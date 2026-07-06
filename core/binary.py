from __future__ import annotations

import os
import struct
from typing import BinaryIO, Tuple


class BinaryReader:
    def __init__(self, fh: BinaryIO, endian: str = "<"):
        if endian not in ("<", ">"):
            raise ValueError(f"Unsupported endianness: {endian}")
        self._fh = fh
        self.endian = endian

    def tell(self) -> int:
        return self._fh.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> None:
        self._fh.seek(offset, whence)

    def skip(self, size: int) -> None:
        self.seek(size, os.SEEK_CUR)

    def read(self, size: int) -> bytes:
        data = self._fh.read(size)
        if len(data) != size:
            raise EOFError(f"Expected {size} bytes at 0x{self.tell():X}, got {len(data)}")
        return data

    def peek(self, size: int, offset: int | None = None) -> bytes:
        previous = self.tell()
        try:
            if offset is not None:
                self.seek(offset)
            return self.read(size)
        finally:
            self.seek(previous)

    def _unpack(self, fmt: str, size: int) -> int | float:
        return struct.unpack(self.endian + fmt, self.read(size))[0]

    def _peek_unpack(self, fmt: str, size: int, offset: int | None = None) -> int | float:
        return struct.unpack(self.endian + fmt, self.peek(size, offset=offset))[0]

    def i8(self) -> int:
        return int(self._unpack("b", 1))

    def u8(self) -> int:
        return int(self._unpack("B", 1))

    def i16(self) -> int:
        return int(self._unpack("h", 2))

    def u16(self) -> int:
        return int(self._unpack("H", 2))

    def i32(self) -> int:
        return int(self._unpack("i", 4))

    def u32(self) -> int:
        return int(self._unpack("I", 4))

    def f32(self) -> float:
        return float(self._unpack("f", 4))

    def vec4(self) -> Tuple[float, float, float, float]:
        return (self.f32(), self.f32(), self.f32(), self.f32())

    def peek_i16(self, offset: int | None = None) -> int:
        return int(self._peek_unpack("h", 2, offset=offset))

    def peek_u8(self, offset: int | None = None) -> int:
        return int(self._peek_unpack("B", 1, offset=offset))

    def peek_u16(self, offset: int | None = None) -> int:
        return int(self._peek_unpack("H", 2, offset=offset))

    def peek_u32(self, offset: int | None = None) -> int:
        return int(self._peek_unpack("I", 4, offset=offset))
