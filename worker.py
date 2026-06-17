import asyncio
async def worker(name: str, queue: asyncio.Queue, model):
    while True:
        try:
            request, future = await queue.get()
            output = model(request.prompt, max_tokens=256)
            result = output["choices"][0]["text"]
            future.set_result(result)
            print(f"Worker {name} finished processing request: {request}")
            queue.task_done()
        except Exception as e:
            print(f"Worker {name} encountered an error: {e}")
