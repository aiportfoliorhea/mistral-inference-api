# mistral-inference-api

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
