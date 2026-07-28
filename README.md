# mistral-inference-api

# Mistral Inference API

LLM inference optimizer built on fine-tuned Mistral 7B. Benchmarks three serving 
frameworks (llama.cpp, FastAPI async, vLLM) on FinanceBench financial QA tasks.
Implements QLoRA fine-tuning, INT4/INT8 quantization, async request queuing, 
and dynamic batching.

## Stack
- **Model:** Mistral 7B Instruct v0.3, fine-tuned with QLoRA on FinanceBench
- **Quantization:** bitsandbytes (INT4, INT8) + llama.cpp GGUF (Q8_0)
- **Serving:** llama.cpp · FastAPI async · vLLM (PagedAttention)
- **Training:** Kaggle T4 GPU · PEFT · FinanceBench instruction format
- **Benchmarking:** Lambda Labs A10 GPU

Dynamic batching variant collects requests for 50ms before firing inference.

## Kaggle Notebooks
- [Fine-tuning + INT8 Quantization notebook](https://www.kaggle.com/code/aiportfoliorhea/int8-ft-final) - QLoRA on FinanceBench, bitsandbytes INT4/INT8
- [Dynamic Batching notebook](https://www.kaggle.com/code/aiportfoliorhea/fast-async-batching) - Batching with FAST API Async
- Serving benchmarks run manually on Lambda Labs A10 GPU using curl and Python timing scripts (not saved as notebook)

## Architecture
Request → FastAPI endpoint → asyncio.Queue → Worker → llama_cpp / vLLM → Response

## Model Quality Benchmarks (FinanceBench F1)

Fine-tuned Mistral 7B on FinanceBench dataset using QLoRA (PEFT).
Evaluated with F1 score against ground truth answers.

| Metric | Base Mistral 7B | Fine-tuned INT4 | Fine-tuned INT8 |
|---|---|---|---|
| F1 (original prompt) | 14.5% | 12.3% | 12.7% |
| F1 (explicit instruction prompt) | 4.2% | 9.0% | **16.9%** |

**Key finding:** Explicit instruction prompt hurt the base model (14.5% → 4.2%) 
but helped both fine-tuned models. The base model wasn't trained to follow terse 
instructions — constraining it breaks its output format. Fine-tuned models learned 
the short-answer format, so the explicit prompt reinforces it rather than fighting it.

**Winner:** Fine-tuned INT8 + explicit instruction prompt at **16.9% F1**.

## Benchmark Results

All benchmarks run on Lambda Labs A10 GPU unless noted.

### Serving Framework Comparison (Mistral 7B, FinanceBench prompts)

| Framework | Single Latency | Throughput | Concurrent 10 |
|---|---|---|---|
| llama.cpp direct | 5.716s | 0.176 req/s | 14.572s |
| FastAPI async | 4.642s | 0.227 req/s | 45.192s |
| vLLM (PagedAttention) | 3.588s | 0.279 req/s | 4.046s |

### Dynamic Batching (FastAPI + 50ms window)

| Metric | Value |
|---|---|
| Single request latency | 18.8s |
| Throughput | 0.054 req/s |
| Concurrent 10 requests | 184s |

> ⚠️ Measured on Kaggle CPU (no CUDA). Not directly comparable to table above.
> Key finding: dynamic batching adds 50ms window overhead with no concurrency gain
> on CPU because llama_cpp holds the GIL — model is the bottleneck, not the server.
> vLLM solves this at the engine level via PagedAttention.

## Key Findings
- vLLM delivers **1.6x throughput** vs llama.cpp direct
- vLLM handles 10 concurrent requests in **4s vs 45s** for FastAPI async
- Dynamic batching only helps when the server is the bottleneck, not the model

## Serving Saturation Sweep (vLLM)

Ran a closed loop concurrency sweep (1 to 128) on vLLM to find the saturation point, 
using a 512/128 token workload on an A10 GPU.

| Concurrency | Output Throughput (tok/s) | p95 TTFT (ms) |
|---|---|---|
| 1 | 29.8 | 123 |
| 2 | 56.2 | 264 |
| 4 | 105.7 | 499 |
| 8 | 183.3 | 920 |
| 16 | 317.4 | 1,460 |
| 32 | 478.8 | 1,970 |
| **64** | **616.0** | **3,810** |
| 128 | 588.7 | 14,957 |

**Key finding:** throughput peaks at 616 tok/s at concurrency 64, then regresses to 
588.7 tok/s at 128, while p95 TTFT jumps from 3.8s to nearly 15s. Prometheus metrics 
confirm the cause. `vllm:kv_cache_usage_perc` climbs from roughly 0.6 to 0.8 at lower 
concurrency to pinned near 0.85 to 1.0 past the saturation point, and 
`vllm:num_requests_waiting` goes from flat at 0 to a sustained queue of about 35 to 40 
requests. The GPU's KV cache runs out of room before compute does, so past 64 
concurrent requests, work queues up instead of processing in parallel. Throughput 
flattens and then drops, and latency compounds.
