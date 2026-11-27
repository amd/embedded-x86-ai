# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""scripts/ensure_voice.sh on a machine set up before Gemma 4.

Each test runs the real script in a scratch repository root: a copy of
``vla_pipeline`` and the script, its own ``config/pipeline.yaml``, this clone's
llama.cpp checkout and llama-server, and this clone's Gemma files linked in.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

from notebook_harness import PIPE_ROOT

pytestmark = pytest.mark.sim

SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
LLAMA_DIR = PIPE_ROOT / "third_party" / "llama.cpp"
REAL_GEMMA = PIPE_ROOT / "models" / "gemma-4-E4B-it"
GEMMA_FILES = ("gemma-4-E4B-it-Q4_0.gguf", "mmproj-gemma-4-E4B-it-Q8_0.gguf")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def root():
    # Not tmp_path: /tmp is mounted noexec here, and the script, the venv
    # python wrapper and the stand-in binaries all have to run.
    scratch = PIPE_ROOT / "_artifacts" / "ensure_voice"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=scratch))
    try:
        yield _build_root(tmp_path)
    finally:
        shutil.rmtree(tmp_path)


def _build_root(tmp_path: Path) -> Path:
    for name in GEMMA_FILES:
        assert (REAL_GEMMA / name).is_file(), f"{REAL_GEMMA / name} is missing - run ./bootstrap.sh"
    assert (LLAMA_DIR / "build" / "bin" / "llama-server").is_file(), "llama.cpp is not built - run ./bootstrap.sh"
    (tmp_path / "scripts").mkdir()
    shutil.copy2(PIPE_ROOT / "scripts" / "ensure_voice.sh", tmp_path / "scripts")
    shutil.copytree(PIPE_ROOT / "vla_pipeline", tmp_path / "vla_pipeline",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "third_party").mkdir()
    (tmp_path / "third_party" / "llama.cpp").symlink_to(LLAMA_DIR)
    gemma = tmp_path / "models" / "gemma-4-E4B-it"
    gemma.mkdir(parents=True)
    for name in GEMMA_FILES:
        (gemma / name).symlink_to(REAL_GEMMA / name)
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(f'#!/bin/sh\nexec "{PIPE_ROOT / ".venv" / "bin" / "python"}" "$@"\n')
    python.chmod(0o755)
    cfg = yaml.safe_load((PIPE_ROOT / "config" / "pipeline.yaml").read_text())
    cfg["voice_intent"] = "gemma4-e4b-it"
    (tmp_path / "config").mkdir()
    _write_config(tmp_path, cfg)
    return tmp_path


def _write_config(root: Path, cfg: dict) -> None:
    (root / "config" / "pipeline.yaml").write_text(yaml.safe_dump(cfg))


def _config(root: Path) -> dict:
    return yaml.safe_load((root / "config" / "pipeline.yaml").read_text())


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(root / "scripts" / "ensure_voice.sh"), *args],
        env={**os.environ, "VENV_DIR": str(root / "venv")},
        capture_output=True, text=True, timeout=120,
    )


def _place_silero(root: Path) -> Path:
    target = root / _config(root)["gemma4-e4b-it"]["vad_model"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PIPE_ROOT / "models" / "silero_vad.onnx", target)
    return target


def test_a_machine_with_every_voice_file_downloads_nothing(root):
    _place_silero(root)
    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert "downloading" not in result.stdout


def test_a_missing_silero_model_is_downloaded_and_checked(root):
    result = _run(root)
    assert result.returncode == 0, result.stderr
    target = root / _config(root)["gemma4-e4b-it"]["vad_model"]
    assert _sha256(target) == SILERO_SHA256
    assert not target.with_name(target.name + ".part").exists()


def test_a_corrupt_silero_model_is_replaced(root):
    target = _place_silero(root)
    target.write_bytes(b"truncated")
    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert _sha256(target) == SILERO_SHA256


def test_a_missing_gemma_file_without_the_hf_cli_fails_and_names_it(root):
    _place_silero(root)
    (root / "models" / "gemma-4-E4B-it" / GEMMA_FILES[1]).unlink()
    result = _run(root)
    assert result.returncode == 1
    assert GEMMA_FILES[1] in result.stderr
    assert "no hf CLI" in result.stderr


def test_a_missing_gemma_file_is_fetched_from_the_gemma_repo(root):
    # hf stands in for the 560 MB download: it records its arguments and
    # writes the file where --local-dir says.
    _place_silero(root)
    (root / "models" / "gemma-4-E4B-it" / GEMMA_FILES[1]).unlink()
    record = root / "hf-args"
    hf = root / "venv" / "bin" / "hf"
    hf.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{record}"\n'
        'while [ $# -gt 0 ]; do [ "$1" = --local-dir ] && dir="$2"; shift; done\n'
        f'touch "$dir/{GEMMA_FILES[1]}"\n'
    )
    hf.chmod(0o755)
    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert record.read_text().split() == [
        "download", "ggml-org/gemma-4-E4B-it-GGUF", GEMMA_FILES[1],
        "--local-dir", str(root / "models" / "gemma-4-E4B-it"),
    ]


def test_a_llama_server_older_than_gemma4_audio_fails(root):
    _place_silero(root)
    old = root / "old-llama-server"
    # 63f8fe0ef (2026-04-02) is ten days before the Gemma 4 audio encoder.
    old.write_text("#!/bin/sh\necho 'version: 0.0.0 (build 8637, commit 63f8fe0ef)'\n")
    old.chmod(0o755)
    cfg = _config(root)
    cfg["gemma4-e4b-it"]["server_bin"] = str(old)
    _write_config(root, cfg)
    result = _run(root)
    assert result.returncode == 1
    assert "predates Gemma 4 audio" in result.stderr


def test_models_only_skips_the_llama_server_check(root):
    _place_silero(root)
    cfg = _config(root)
    cfg["gemma4-e4b-it"]["server_bin"] = str(root / "not-built" / "llama-server")
    _write_config(root, cfg)
    assert _run(root).returncode == 1
    result = _run(root, "--models-only")
    assert result.returncode == 0, result.stderr
