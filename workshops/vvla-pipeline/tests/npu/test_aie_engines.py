# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""AI Engine counts from xrt-smi reports. No device and no xrt-smi."""

from __future__ import annotations

import struct

from vla_pipeline.utils.aie_engines import (
    decode_aie_metadata,
    partitions_from_json,
    sample_from,
    summarize,
    topology_from_text,
    total_columns,
)

# This machine, from DRM_AMDXDNA_QUERY_AIE_METADATA: 4 core rows, 1 mem row, 1 shim row, 8 columns.
_HALO = {
    "columns": 8,
    "total_rows": 6,
    "core_rows": 4,
    "core_row_start": 2,
    "mem_rows": 1,
    "mem_row_start": 1,
    "shim_rows": 1,
    "shim_row_start": 0,
}

_DEVICE_TABLE = """\
Device(s) Present
|BDF             |Name          |Architecture  |Topology  |
|----------------|--------------|--------------|----------|
|[0000:c4:00.1]  |RyzenAI-npu5  |aie2p         |6x8       |
"""


def _part(start, count, pid, completions):
    return {
        "start_col": str(start),
        "num_cols": str(count),
        "hw_contexts": [{
            "pid": str(pid),
            "slot_index": "0",
            "command_completions": str(completions),
        }],
    }


def _report(*parts):
    return {"devices": [{"aie_partitions": {"partitions": list(parts)}}]}


def test_topology_counts_every_tile_row():
    """xrt-smi's 6x8 is shim + memory + core, not 6 engines per column."""
    assert topology_from_text(_DEVICE_TABLE) == (6, 8)
    assert total_columns({"devices": [{"platforms": [{"static_region": {"total_columns": "8"}}]}]}) == 8


def test_driver_metadata_is_four_core_rows():
    raw = bytearray(64)
    struct.pack_into("<HH", raw, 4, 8, 6)
    struct.pack_into("<HH", raw, 16, 4, 2)
    struct.pack_into("<HH", raw, 32, 1, 1)
    struct.pack_into("<HH", raw, 48, 1, 0)
    assert decode_aie_metadata(bytes(raw)) == _HALO


def test_empty_partition_string_is_no_context():
    data = {"devices": [{"aie_partitions": {"partitions": ""}}]}
    assert partitions_from_json(data) == []
    sample = sample_from(_HALO, [], None)
    assert sample["engines_reserved"] == 0
    assert sample["memory_tiles_reserved"] == 0
    assert sample["engines_busy"] is None


def test_one_full_array_and_two_halves():
    one = partitions_from_json(_report(_part(0, 8, 10, 1)))
    first = sample_from(_HALO, one, None)
    assert first["engines_reserved"] == 32
    assert first["memory_tiles_reserved"] == 8
    assert first["engines_busy"] is None
    second = sample_from(_HALO, partitions_from_json(_report(_part(0, 8, 10, 4))), first)
    assert second["engines_busy"] == 32

    halves = partitions_from_json(_report(_part(0, 4, 10, 2), _part(4, 4, 11, 2)))
    split = sample_from(_HALO, halves, None)
    assert split["engines_reserved"] == 32
    assert len(split["partitions"]) == 2
    assert [p["num_cols"] for p in split["partitions"]] == [4, 4]
    lines = summarize([split, sample_from(_HALO, halves, split)])
    assert any("32 AI engines and 8 memory tiles" in line for line in lines)
    assert any("peak 2 context(s) in 2 partition(s)" in line for line in lines)
    assert any(
        "columns 0-3, 16 AI engines, 4 memory tiles | columns 4-7, 16 AI engines, 4 memory tiles" in line
        for line in lines
    )


def test_idle_context_is_reserved_and_not_busy():
    held = partitions_from_json(_report(_part(0, 8, 10, 5)))
    first = sample_from(_HALO, held, None)
    second = sample_from(_HALO, held, first)
    assert second["engines_reserved"] == 32
    assert second["engines_busy"] == 0
