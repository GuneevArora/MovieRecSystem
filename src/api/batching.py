"""
Dynamic micro-batching for model inference.

Running one forward pass per request does not scale: concurrent requests
fight over the same CPU cores (or GPU) and tail latency explodes. Instead a
single worker drains whatever requests are waiting and runs them as one
batch. When the service is idle a request is served immediately as a batch
of one, so batching adds no latency at low load; under load, batches grow
naturally because requests queue up while the previous batch is running.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class MicroBatcher:
    def __init__(self, batch_fn: Callable[[list], list], max_batch_size: int = 64):
        """
        batch_fn: takes a list of items and returns one result per item, in
        order. It runs on a dedicated thread, never on the event loop.
        """
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.batches_run = 0
        self.items_run = 0

    async def start(self):
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())

    async def stop(self):
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    async def submit(self, item: Any) -> Any:
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((item, future))
        return await future

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            items = [item for item, _ in batch]
            try:
                results = await loop.run_in_executor(self._executor, self.batch_fn, items)
                for (_, future), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as error:  # surface the failure to every waiting request
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)

            self.batches_run += 1
            self.items_run += len(batch)
