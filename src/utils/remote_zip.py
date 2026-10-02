"""Read selected members of a large HTTP ZIP using verified byte ranges."""

from __future__ import annotations

import io
from urllib.request import Request, urlopen


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, block_size: int = 4 * 1024 * 1024):
        self.url = url
        self.block_size = block_size
        with urlopen(Request(url, method="HEAD"), timeout=60) as response:
            if response.headers.get("Accept-Ranges") != "bytes":
                raise ValueError("Server does not advertise HTTP byte ranges")
            self.length = int(response.headers["Content-Length"])
        self.position = 0
        self.cache = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_END:
            offset += self.length
        elif whence == io.SEEK_CUR:
            offset += self.position
        if offset < 0:
            raise ValueError("Negative seek position")
        self.position = offset
        return offset

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.length - self.position
        end = min(self.length, self.position + size)
        chunks = []
        while self.position < end:
            block_index = self.position // self.block_size
            if block_index not in self.cache:
                start = block_index * self.block_size
                stop = min(self.length, start + self.block_size) - 1
                request = Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
                with urlopen(request, timeout=120) as response:
                    expected = f"bytes {start}-{stop}/{self.length}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise IOError("Server did not honor the requested byte range")
                    block = response.read()
                if len(block) != stop - start + 1:
                    raise IOError("Incomplete HTTP byte range")
                self.cache[block_index] = block
            block = self.cache[block_index]
            start_in_block = self.position % self.block_size
            amount = min(end - self.position, len(block) - start_in_block)
            chunks.append(block[start_in_block:start_in_block + amount])
            self.position += amount
        return b"".join(chunks)
