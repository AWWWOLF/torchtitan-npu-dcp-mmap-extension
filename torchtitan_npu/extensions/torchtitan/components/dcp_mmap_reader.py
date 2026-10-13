# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in mmap reader for local DCP ``torch.save`` blobs.

This experiment deliberately centralizes all private PyTorch serialization API
usage in this module. Unsupported blob layouts fall back to the native reader
path for that item; malformed metadata and unsafe physical ranges fail closed.
"""

import io
import os
import pickle
import struct
import sys
from collections import Counter
from typing import Any, cast, IO

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemReader
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner
from torch.futures import Future
from torch.serialization import (
    _is_torchscript_zip,
    _is_zipfile,
    _load,
    _open_zipfile_reader,
)


class _UnsupportedMmapBlob(RuntimeError):
    """The blob is valid but cannot safely use mapped tensor storage."""


def _load_mapped_blob(
    blob_view: IO[bytes],
    overall_storage: torch.UntypedStorage,
    base_offset: int,
    region_length: int,
) -> Any:
    if base_offset < 0 or region_length <= 0:
        raise RuntimeError("invalid mapped DCP blob range")
    if base_offset + region_length > overall_storage.nbytes():
        raise RuntimeError("mapped DCP blob range exceeds its backing file")
    if not blob_view.seekable():
        raise _UnsupportedMmapBlob("blob stream is not seekable")

    original_position = blob_view.tell()
    try:
        blob_view.seek(0, os.SEEK_END)
        if blob_view.tell() != region_length:
            raise RuntimeError("DCP metadata length does not match the blob view")
        blob_view.seek(0)
        if not _is_zipfile(blob_view):
            raise _UnsupportedMmapBlob("blob is not a torch.save ZIP archive")

        blob_view.seek(0)
        with _open_zipfile_reader(blob_view) as zip_file:
            if _is_torchscript_zip(zip_file):
                raise _UnsupportedMmapBlob("TorchScript archives are unsupported")
            if zip_file.has_record("byteorder"):
                byteorder = zip_file.get_record("byteorder")
                if byteorder not in (b"little", b"big"):
                    raise RuntimeError(f"unknown checkpoint byte order: {byteorder!r}")
                if byteorder.decode() != sys.byteorder:
                    raise _UnsupportedMmapBlob("cross-endian blobs are unsupported")

            get_header_offset = getattr(zip_file, "get_record_header_offset", None)
            if get_header_offset is None:
                raise _UnsupportedMmapBlob("ZIP reader lacks local-header offsets")
            records = [n for n in zip_file.get_all_records() if n.startswith("data/")]
            if not records:
                raise _UnsupportedMmapBlob("blob has no tensor storage records")
            for name in records:
                header_offset = get_header_offset(name)
                record_offset = zip_file.get_record_offset(name)
                if not isinstance(header_offset, int) or header_offset < 0:
                    raise RuntimeError(f"invalid ZIP header offset for {name!r}")
                blob_view.seek(header_offset)
                header = blob_view.read(30)
                if len(header) != 30 or header[:4] != b"PK\x03\x04":
                    raise RuntimeError(f"invalid ZIP local header for {name!r}")
                flags, compression = struct.unpack_from("<HH", header, 6)
                if flags & 0x1 or compression != 0:
                    raise _UnsupportedMmapBlob(
                        f"ZIP record {name!r} is encrypted or compressed"
                    )
                filename_length, extra_length = struct.unpack_from("<HH", header, 26)
                expected = header_offset + 30 + filename_length + extra_length
                if expected != record_offset or not 0 <= record_offset < region_length:
                    raise RuntimeError(f"inconsistent ZIP record offset for {name!r}")

            region_storage = overall_storage[base_offset : base_offset + region_length]
            return _load(
                zip_file,
                map_location="cpu",
                pickle_module=_weights_only_unpickler,
                overall_storage=region_storage,
                weights_only=True,
            )
    finally:
        blob_view.seek(original_position)


class MmapFileSystemReader(FileSystemReader):
    """Local DCP reader that maps each backing file once, with native fallback."""

    def __init__(self, path: str | os.PathLike) -> None:
        super().__init__(path)
        self._mapped_storages: dict[str, torch.UntypedStorage] = {}
        self.mapped_files: set[str] = set()
        self.mmap_hit_requests = 0
        self.fallback_requests = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.mapped_virtual_bytes = 0

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        super().reset(checkpoint_id)
        self._mapped_storages.clear()
        self.mapped_files.clear()
        self.mmap_hit_requests = 0
        self.fallback_requests = 0
        self.fallback_reasons.clear()
        self.mapped_virtual_bytes = 0

    def _fallback_tensor(self, stream: IO[bytes]) -> Tensor:
        if stream.seekable():
            stream.seek(0)
            seekable = stream
        else:
            seekable = io.BytesIO(stream.read(-1))
            seekable.seek(0)
        return cast(
            Tensor,
            torch.load(seekable, map_location="cpu", weights_only=True),
        )

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: dict[str, list[Any]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            per_file.setdefault(item_md.relative_path, []).append(read_item)

        for relative_path, requests in per_file.items():
            path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(path, "rb") as stream:
                for request in requests:
                    item_md = self.storage_data[request.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    descriptors = item_md.transform_descriptors or ()
                    transformed = self.transforms.transform_load_stream(
                        request, descriptors, file_slice
                    )
                    if request.type == LoadItemType.BYTE_IO:
                        value = io.BytesIO(transformed.read(-1))
                        value.seek(0)
                        planner.load_bytes(request, value)
                        continue

                    tensor = None
                    if type(self.fs) is FileSystem and not descriptors:
                        physical_path = os.fspath(path)
                        file_size = os.path.getsize(physical_path)
                        if (
                            item_md.offset < 0
                            or item_md.length <= 0
                            or item_md.offset + item_md.length > file_size
                        ):
                            raise RuntimeError(
                                "DCP storage range is outside the backing file: "
                                f"offset={item_md.offset}, length={item_md.length}, "
                                f"file_size={file_size}"
                            )
                        storage = self._mapped_storages.get(physical_path)
                        if storage is None:
                            storage = torch.UntypedStorage.from_file(
                                physical_path, shared=False, nbytes=file_size
                            )
                            self._mapped_storages[physical_path] = storage
                            self.mapped_files.add(physical_path)
                            self.mapped_virtual_bytes += file_size
                        try:
                            tensor = _load_mapped_blob(
                                file_slice, storage, item_md.offset, item_md.length
                            )
                            self.mmap_hit_requests += 1
                        except _UnsupportedMmapBlob as error:
                            self.fallback_requests += 1
                            self.fallback_reasons[str(error)] += 1
                            file_slice.seek(0)
                    else:
                        self.fallback_requests += 1
                        reason = "non-local filesystem" if type(self.fs) is not FileSystem else "stream transform"
                        self.fallback_reasons[reason] += 1

                    if tensor is None:
                        tensor = self._fallback_tensor(transformed)
                    tensor = narrow_tensor_by_index(
                        cast(Tensor, tensor), request.storage_offsets, request.lengths
                    )
                    target = planner.resolve_tensor(request).detach()
                    if target.size() != tensor.size():
                        raise AssertionError(
                            f"request {request.storage_index} size mismatch: "
                            f"{target.size()} != {tensor.size()}"
                        )
                    target.copy_(tensor)
                    planner.commit_tensor(request, target)

        future: Future = Future()
        future.set_result(None)
        return future
