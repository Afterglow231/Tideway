"""One bounded SSD batch in flight; payload storage belongs to the RAM tier.

Only the inference thread changes cache metadata. The worker writes destinations
whose row leases remain held until ``finish`` succeeds. Callers must finish the
batch before using the reader synchronously: its O_DIRECT bounce buffers are
shared between calls, even though each call uses several I/O lanes.
"""

from concurrent.futures import ThreadPoolExecutor


class AsyncReadBatch:
    def __init__(self, reader):
        self.reader = reader
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tideway-prefetch")
        self.future = None

    def submit(self, items):
        if self.future is not None:
            raise RuntimeError("SSD prefetch backpressure: previous batch is still owned")
        items = tuple(items)
        if not items:
            raise ValueError("cannot submit an empty SSD prefetch")
        self.future = self.executor.submit(self._read, items)

    def _read(self, items):
        try:
            self.reader.read_many(items)
        except BaseException:
            # DirectExperts' map may raise while another lane still writes RAM.
            # This is a fatal I/O failure: drain all lanes before propagating it.
            pool = getattr(self.reader, "pool", None)
            if pool is not None:
                pool.shutdown(wait=True)
            raise

    def finish(self):
        if self.future is None:
            return
        self.future.result()
        self.future = None

    def close(self):
        try:
            self.finish()
        finally:
            self.executor.shutdown(wait=True)
