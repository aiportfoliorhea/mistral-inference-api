from unittest import result

from llama_cpp import Llama
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel


import asyncio

from worker import worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Llama(model_path="/Users/rheaupadhyay/mistral-inference-api/models/mistral_finetuned_q8.gguf", n_threads=2, n_gpu_layers=-1)
    queue = asyncio.Queue()
    app.state.job_queue = queue
    worker_task = asyncio.create_task(worker("Worker-1", queue, model=
                                             app.state.model))
    yield
    worker_task.cancel()

app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: Request):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await app.state.job_queue.put((request, future))
    result = await future
    return {"response": result}

    
    