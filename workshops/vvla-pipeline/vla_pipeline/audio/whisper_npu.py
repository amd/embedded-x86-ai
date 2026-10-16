# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Whisper ASR on the Ryzen AI NPU - based on the AMD RyzenAI-SW Whisper demo.

Core inference (``WhisperONNX``) is taken from RyzenAI-SW ``demo/ASR/Whisper``:
fixed-length greedy decode against NPU-optimized encoder/decoder ONNX models
downloaded from the ``amd/whisper-*-onnx-npu`` Hugging Face repos, with
per-model VitisAI compiler configs
(``config/vitisai_config_whisper_{encoder,decoder}.json``).

``WhisperNPU`` wraps it with the pipeline config and a ``transcribe`` method
that ``vla_pipeline.main`` drives through
:class:`~vla_pipeline.audio.vad_listener.FastListener`; its ``listen_once``
gives the standalone CLI below its own single-utterance mic capture.

Standalone test:

    python -m vla_pipeline.audio.whisper_npu --input mic --device npu
    python -m vla_pipeline.audio.whisper_npu --input some.wav --device cpu
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from vla_pipeline.utils.config import load_config, resolve
from vla_pipeline.utils.npu_session import size_npu

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 0.1 sec chunks

# NPU-optimized ONNX published by AMD (RyzenAI-SW ASR demo). Each repo lays out
# its encoder/decoder under different filenames; the matching HF tokenizer/
# feature-extractor must also be the right variant. AMD's *base* NPU model is
# the ENGLISH-ONLY checkpoint, so its tokenizer is openai/whisper-base.en -
# loading the multilingual openai/whisper-base instead yields mismatched token
# IDs and empty/garbage transcripts.
HF_MODEL_MAP = {
    "whisper-base": {
        "repo": "amd/whisper-base-en-onnx-npu",
        "encoder": "base_en_encoder.onnx",
        "decoder": "base_en_decoder.onnx",
        "tokenizer": "openai/whisper-base.en",
        "english_only": True,
    },
    "whisper-small": {
        "repo": "amd/whisper-small-onnx-npu",
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model.onnx",
        "tokenizer": "openai/whisper-small",
        "english_only": False,
    },
    "whisper-medium": {
        "repo": "amd/whisper-medium-onnx-npu",
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model.onnx",
        "tokenizer": "openai/whisper-medium",
        "english_only": False,
    },
    "whisper-large-v3-turbo": {
        "repo": "amd/whisper-large-turbo-onnx-npu",
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model.onnx",
        "tokenizer": "openai/whisper-large-v3-turbo",
        "english_only": False,
    },
}


# =============================================================================
# AMD RyzenAI-SW WhisperONNX (verbatim inference logic)
# =============================================================================


class WhisperONNX:
    """ONNX Runtime wrapper for the Whisper encoder/decoder, adapted verbatim from the AMD RyzenAI-SW ASR demo."""

    def __init__(
        self,
        encoder_path,
        decoder_path,
        model_type,
        encoder_providers=None,
        decoder_providers=None,
        language=None,
        tokenizer_id=None,
        english_only=False,
    ):
        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=decoder_providers)

        # Tokenizer/feature-extractor MUST match the model checkpoint variant.
        # AMD's base NPU model is English-only -> openai/whisper-base.en.
        # Prefer the local cache: an expired Hugging Face token returns 401
        # even when these public files are already downloaded.
        tok_id = tokenizer_id or f"openai/{model_type}"
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                tok_id, local_files_only=True
            )
            self.tokenizer = WhisperTokenizer.from_pretrained(tok_id, local_files_only=True)
        except OSError:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tok_id)
            self.tokenizer = WhisperTokenizer.from_pretrained(tok_id)
        self.decoder_start_token = self.sot_token = (
            self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        )
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = min(448, self.decoder.get_inputs()[0].shape[1])
        if not isinstance(self.max_length, int):
            raise ValueError("Invalid/Dynamic input shapes")

        self.english_only = english_only
        self.language = None if english_only else language
        if english_only:
            # English-only Whisper has no language/task selector tokens; the
            # prefix is just the start-of-transcript (+ notimestamps).
            self.initial_tokens = list(self.tokenizer.prefix_tokens)
        elif self.language:
            self.tokenizer.set_prefix_tokens(language=self.language, task="transcribe")
            self.initial_tokens = list(self.tokenizer.prefix_tokens)
        else:
            self.initial_tokens = [self.decoder_start_token]

    def preprocess(self, audio):
        """Convert raw audio to Whisper log-mel spectrogram."""
        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        )
        return inputs["input_features"]

    def encode(self, input_features):
        """Run encoder ONNX model."""
        input_name = self.encoder.get_inputs()[0].name
        return self.encoder.run(None, {input_name: input_features})[0]

    def decode(self, encoder_out):
        """Greedy decode with fixed-length input_ids."""
        tokens = list(self.initial_tokens)
        first_token_delay = None
        decode_start = time.time()

        # Get decoder input names
        decoder_inputs = self.decoder.get_inputs()
        input_ids_name = decoder_inputs[0].name
        encoder_out_name = decoder_inputs[1].name

        # Distinguish inputs by data type if the order is not guaranteed
        if decoder_inputs[0].type != "tensor(int64)":
            input_ids_name, encoder_out_name = encoder_out_name, input_ids_name

        for _ in range(len(tokens), self.max_length):
            decoder_input = np.full(
                (1, self.max_length), self.eos_token, dtype=np.int64
            )
            decoder_input[0, : len(tokens)] = tokens

            outputs = self.decoder.run(
                None, {input_ids_name: decoder_input, encoder_out_name: encoder_out}
            )
            logits = outputs[0]
            next_token = int(np.argmax(logits[0, len(tokens) - 1]))

            if next_token == self.eos_token:
                break
            tokens.append(next_token)
            if first_token_delay is None:
                first_token_delay = time.time() - decode_start
        return tokens, first_token_delay

    def transcribe(self, audio, chunk_length_s=30, is_mic=False):
        """Full encode-decode pipeline with chunked long-form support."""
        chunk_size = SAMPLE_RATE * chunk_length_s
        total_samples = len(audio)
        transcription = []
        chunk_idx = 0
        total_start_time = time.time()

        overlap = SAMPLE_RATE * 1
        for start in range(0, total_samples, chunk_size - overlap):
            end = min(start + chunk_size, total_samples)
            audio_chunk = audio[start:end]

            input_features = self.preprocess(audio_chunk)
            encoder_out = self.encode(input_features)
            tokens, first_token_delay = self.decode(encoder_out)
            decoded_text = self.tokenizer.decode(
                tokens[len(self.initial_tokens) :], skip_special_tokens=True
            ).strip()
            transcription.append(decoded_text)
            chunk_idx += 1
            if not is_mic:
                if first_token_delay is not None:
                    print(f"\nPerformance Metric (Chunk {chunk_idx}):")
                    print(
                        f" Time to First Token for this chunk: {first_token_delay:.2f} seconds"
                    )
                else:
                    print(f"\nPerformance Metric (Chunk {chunk_idx}):")
                    print(
                        " Time to First Token for this chunk: n/a (no token before EOS)"
                    )

        total_end_time = time.time()
        input_audio_duration = total_samples / SAMPLE_RATE
        rtf = (total_end_time - total_start_time) / input_audio_duration
        if not is_mic:
            print(f" RTF: {rtf:.2f}")

        return " ".join(transcription), rtf


# =============================================================================
# Model download / provider options (AMD demo logic, driven by pipeline.yaml)
# =============================================================================


def download_whisper_onnx(
    model_type: str, dest_dir: str | Path | None = None
) -> tuple[str, str]:
    """Download NPU-optimized Whisper ONNX from Hugging Face if needed.

    Fetches only the encoder/decoder ONNX files by exact name (not a full
    snapshot), so it won't stall pulling extra repo artifacts. When ``dest_dir``
    is given, the files are placed there (e.g. ``models/whisper-base/``) instead
    of the shared HF cache; the canonical encoder/decoder names are used so the
    on-disk layout is stable regardless of the repo's own filenames.
    """
    from huggingface_hub import hf_hub_download

    entry = HF_MODEL_MAP.get(model_type)
    if entry is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for ONNX auto-download. "
            f"Known: {sorted(HF_MODEL_MAP)}"
        )

    repo_id = entry["repo"]
    if dest_dir is not None:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        encoder_path = str(dest / "encoder_model.onnx")
        decoder_path = str(dest / "decoder_model.onnx")
        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            # hf_hub_download caches then we copy to the canonical local name.
            import shutil

            enc_cached = hf_hub_download(repo_id=repo_id, filename=entry["encoder"])
            dec_cached = hf_hub_download(repo_id=repo_id, filename=entry["decoder"])
            shutil.copyfile(enc_cached, encoder_path)
            shutil.copyfile(dec_cached, decoder_path)
    else:
        encoder_path = hf_hub_download(repo_id=repo_id, filename=entry["encoder"])
        decoder_path = hf_hub_download(repo_id=repo_id, filename=entry["decoder"])

    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
        raise FileNotFoundError(f"Could not find encoder/decoder for {repo_id}")
    return encoder_path, decoder_path


def download_whisper_tokenizer(
    model_type: str, tokenizer_id: str | None = None
) -> str:
    """Cache the Whisper tokenizer and feature extractor used at runtime.

    The ONNX encoder/decoder do not include them. ``WhisperONNX`` and
    ``workshop.common.speech.Whisper`` load this repo with
    ``local_files_only`` first, so bootstrap must place it in the Hugging
    Face cache before an offline run.
    """
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    entry = HF_MODEL_MAP.get(model_type)
    if tokenizer_id:
        tok_id = tokenizer_id
    elif isinstance(entry, dict) and entry.get("tokenizer"):
        tok_id = entry["tokenizer"]
    else:
        tok_id = f"openai/{model_type}"

    def _load(cls):
        try:
            return cls.from_pretrained(tok_id, local_files_only=True)
        except OSError:
            return cls.from_pretrained(tok_id)

    _load(WhisperFeatureExtractor)
    _load(WhisperTokenizer)
    return tok_id


def build_provider_opts(opts: dict) -> list:
    """AMD demo provider-option builder: VitisAI EP if a config_file is set."""
    if opts.get("config_file"):
        return [
            (
                "VitisAIExecutionProvider",
                {
                    "config_file": opts["config_file"],
                    "cache_dir": opts.get("cache_dir", ""),
                    "cache_key": opts.get("cache_key", ""),
                },
            )
        ]
    return ["CPUExecutionProvider"]


# =============================================================================
# Non-blocking single-key reader (for "press q to end utterance")
# =============================================================================


class _KeyReader:
    """Non-blocking detector for a single keypress on an interactive TTY.

    Puts the terminal in cbreak (raw-ish) mode so a single character registers
    without Enter, and restores it on stop(). No-op if stdin isn't a TTY (e.g.
    piped input, non-interactive runs), so it degrades safely.
    """

    def __init__(self, key: str = "q"):
        self.key = (key or "q")[:1].lower()
        self._enabled = False
        self._old = None
        self._fd = None

    def start(self) -> None:
        """Enable cbreak mode on the controlling TTY, if there is one."""
        import sys

        if not sys.stdin or not sys.stdin.isatty():
            return  # not interactive - silently disabled
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except Exception:
            self._enabled = False  # e.g. Windows / no termios

    def pressed(self) -> bool:
        """Return True if the configured stop key has been pressed since the last call."""
        if not self._enabled:
            return False
        import select
        import sys

        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            ch = sys.stdin.read(1)
            return ch.lower() == self.key
        return False

    def stop(self) -> None:
        """Restore the terminal's previous mode."""
        if self._enabled and self._old is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
        self._enabled = False


# =============================================================================
# Pipeline wrapper
# =============================================================================


class WhisperNPU:
    """Pipeline-facing wrapper: config-driven setup + one-utterance mic capture."""

    def __init__(self, cfg: dict | None = None, device: str | None = None):
        cfg = cfg or load_config()
        w = cfg["whisper"]
        a = cfg.get("audio", {})
        self.device = (device or w.get("device", "npu")).lower()
        self.silence_threshold = float(a.get("silence_threshold", 0.015))
        self.silence_duration_s = float(a.get("silence_duration_s", 1.2))
        self.max_utterance_s = float(a.get("max_utterance_s", 12.0))
        self.mic_device = a.get("mic_device")

        model_type = w.get("model_type", "whisper-base")
        enc_path = w.get("encoder_onnx")
        dec_path = w.get("decoder_onnx")
        if (
            enc_path
            and dec_path
            and resolve(enc_path).exists()
            and resolve(dec_path).exists()
        ):
            encoder_path, decoder_path = str(resolve(enc_path)), str(resolve(dec_path))
        else:
            _entry = HF_MODEL_MAP.get(model_type, {})
            models_dir = cfg.get("system", {}).get("models_dir", "models")
            dest_dir = resolve(models_dir) / model_type
            logger.info(
                "Downloading NPU-optimized %s from Hugging Face (%s) → %s ...",
                model_type,
                _entry.get("repo", "?") if isinstance(_entry, dict) else "?",
                dest_dir,
            )
            encoder_path, decoder_path = download_whisper_onnx(
                model_type, dest_dir=dest_dir
            )

        if self.device == "npu":
            size_npu(cfg, device)
            cache_dir = str(resolve(w.get("cache_dir", "cache")))
            enc_opts = {
                "config_file": str(
                    resolve(
                        w.get(
                            "encoder_vitisai_config",
                            "config/vitisai_config_whisper_encoder.json",
                        )
                    )
                ),
                "cache_dir": cache_dir,
                "cache_key": w.get(
                    "cache_key_encoder", f"{model_type.replace('-', '_')}_encoder"
                ),
            }
            dec_opts = {
                "config_file": str(
                    resolve(
                        w.get(
                            "decoder_vitisai_config",
                            "config/vitisai_config_whisper_decoder.json",
                        )
                    )
                ),
                "cache_dir": cache_dir,
                "cache_key": w.get(
                    "cache_key_decoder", f"{model_type.replace('-', '_')}_decoder"
                ),
            }
        else:
            enc_opts, dec_opts = {}, {}

        encoder_providers = build_provider_opts(enc_opts)
        decoder_providers = build_provider_opts(dec_opts)
        logger.info(
            "Whisper requested providers - encoder: %s decoder: %s",
            encoder_providers,
            decoder_providers,
        )
        if (
            self.device == "npu"
            and "VitisAIExecutionProvider" not in ort.get_available_providers()
        ):
            logger.warning(
                "VitisAIExecutionProvider not available in this onnxruntime - "
                "falling back to CPU EP (install the Ryzen AI wheels for NPU)."
            )
            encoder_providers = decoder_providers = ["CPUExecutionProvider"]

        _map_entry = HF_MODEL_MAP.get(model_type, {})
        self.model = WhisperONNX(
            encoder_path,
            decoder_path,
            model_type,
            encoder_providers=encoder_providers,
            decoder_providers=decoder_providers,
            language=w.get("language"),
            tokenizer_id=w.get("tokenizer") or _map_entry.get("tokenizer"),
            english_only=_map_entry.get("english_only", False),
        )
        enc_active = self.model.encoder.get_providers()
        dec_active = self.model.decoder.get_providers()
        logger.info(
            "Whisper active providers - encoder: %s decoder: %s",
            enc_active,
            dec_active,
        )
        if self.device == "npu" and (
            "VitisAIExecutionProvider" not in enc_active
            or "VitisAIExecutionProvider" not in dec_active
        ):
            logger.warning(
                "Whisper requested the NPU but one or both sessions fell back "
                "to CPU; run scripts/verify_npu_stack.py --preflight."
            )

    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one already-captured utterance."""
        text, _rtf = self.model.transcribe(audio, is_mic=True)
        return text.strip()

    def listen_once(
        self,
        silence_threshold: float | None = None,
        silence_duration_s: float | None = None,
        max_utterance_s: float | None = None,
        mic_device: int | None = None,
        on_level=None,
        stop_key: str | None = None,
    ) -> str | None:
        """Block until one utterance is captured; return its transcript.

        Waits for speech (RMS above threshold), records until
        ``silence_duration_s`` of silence, then transcribes. Returns ``None``
        on timeout with no speech or microphone failure.

        ``on_level``: optional callable invoked each audio block with
        ``(rms, speaking, silence_remaining)`` for live UI feedback (e.g. a
        terminal VU meter). ``silence_remaining`` is ``None`` until speech has
        started, then counts down from ``silence_duration_s``.

        ``stop_key``: optional single character (e.g. ``"q"``) that, when
        pressed, immediately ends the utterance and transcribes whatever has
        been captured so far. Requires an interactive TTY; ignored otherwise.
        """
        import sounddevice as sd  # lazy: not needed for WAV mode

        silence_threshold = (
            silence_threshold
            if silence_threshold is not None
            else self.silence_threshold
        )
        silence_duration_s = (
            silence_duration_s
            if silence_duration_s is not None
            else self.silence_duration_s
        )
        max_utterance_s = (
            max_utterance_s if max_utterance_s is not None else self.max_utterance_s
        )
        mic_device = mic_device if mic_device is not None else self.mic_device

        key_reader = _KeyReader(stop_key) if stop_key else None

        q_audio: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, t, status):
            """sounddevice callback: push each captured block onto q_audio."""
            if status:
                logger.warning("sounddevice status: %s", status)
            q_audio.put(indata.copy())

        speech: list[np.ndarray] = []
        speaking = False
        silence_start: float | None = None
        t_start = time.time()
        stopped_by_key = False

        try:
            if key_reader:
                key_reader.start()
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=callback,
                device=mic_device,
            ):
                while True:
                    if key_reader and key_reader.pressed():
                        stopped_by_key = True
                        break
                    if time.time() - t_start > max_utterance_s + 10.0:
                        return None  # nobody spoke
                    try:
                        block = q_audio.get(timeout=0.2).squeeze()
                    except queue.Empty:
                        if on_level is not None:
                            on_level(0.0, speaking, None)
                        continue
                    rms = float(np.sqrt(np.mean(block**2)))
                    if not speaking:
                        if on_level is not None:
                            on_level(rms, False, None)
                        if rms >= silence_threshold:
                            speaking = True
                            speech.append(block)
                        continue
                    speech.append(block)
                    if rms < silence_threshold:
                        silence_start = silence_start or time.time()
                        remaining = silence_duration_s - (time.time() - silence_start)
                        if on_level is not None:
                            on_level(rms, True, max(0.0, remaining))
                        if time.time() - silence_start >= silence_duration_s:
                            break
                    else:
                        silence_start = None
                        if on_level is not None:
                            on_level(rms, True, None)
                    if sum(len(b) for b in speech) >= SAMPLE_RATE * max_utterance_s:
                        break
        except Exception as e:  # PortAudioError etc.
            logger.error("Microphone capture failed: %s", e)
            return None
        finally:
            if key_reader:
                key_reader.stop()

        if not speech:
            return None
        audio = np.concatenate(speech)
        t0 = time.perf_counter()
        text = self.transcribe(audio)
        logger.info(
            "Transcribed %.1fs audio in %.0f ms: %r",
            len(audio) / SAMPLE_RATE,
            (time.perf_counter() - t0) * 1e3,
            text,
        )
        return text or None


# =============================================================================
# Standalone component test
# =============================================================================


def _load_wav(path: str) -> np.ndarray:
    """Load a WAV file and resample it to SAMPLE_RATE if needed.

    soundfile is already a workshop dependency. torchaudio.load() in current
    torchaudio builds requires the separate torchcodec package, which this
    environment does not install.
    """
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE and len(audio):
        new_len = int(round(len(audio) * SAMPLE_RATE / sr))
        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)
    return audio


def _clear_meter_line() -> None:
    """Erase the current terminal line (used to redraw the VU meter in place)."""
    import sys

    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def _make_vu_meter(threshold: float, width: int = 32):
    """Return an on_level callback that draws a live terminal VU meter.

    Shows a bar scaled so the silence threshold sits at a fixed mark, a
    LISTENING/●REC state, the current RMS, and a silence countdown while
    waiting for the utterance to end.
    """
    import sys

    # Scale: map RMS up to ~8x threshold across the bar; mark threshold position.
    full_scale = max(threshold * 8.0, 1e-4)
    thr_pos = int((threshold / full_scale) * width)

    def on_level(rms: float, speaking: bool, silence_remaining):
        """Render one frame of the terminal VU meter for the given RMS/speaking state."""
        filled = int(min(rms / full_scale, 1.0) * width)
        bar = []
        for i in range(width):
            if i < filled:
                bar.append("█")
            elif i == thr_pos:
                bar.append("┊")  # threshold marker
            else:
                bar.append(" ")
        bar_str = "".join(bar)

        if speaking and silence_remaining is not None:
            state = f"\033[33m●REC\033[0m  silence in {silence_remaining:0.1f}s"
        elif speaking:
            state = "\033[31m●REC\033[0m            "
        else:
            state = "\033[36mlistening\033[0m       "

        # Color the bar green when above threshold, dim when below.
        color = "\033[32m" if rms >= threshold else "\033[90m"
        line = f"\r  [{color}{bar_str}\033[0m] {rms:6.4f}  {state}"
        sys.stdout.write(line)
        sys.stdout.flush()

    return on_level


def main() -> None:
    """CLI entry point: transcribe a WAV file or run a live microphone loop."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="'mic' or path to a .wav file")
    parser.add_argument(
        "--device",
        choices=["npu", "cpu"],
        default=None,
        help="override whisper.device from config/pipeline.yaml",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="force decoder language (e.g. 'en'); overrides config",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="mic mode: keep listening for utterances until Ctrl-C",
    )
    parser.add_argument(
        "--no-meter",
        action="store_true",
        help="disable the live mic level meter in mic mode",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.language:
        cfg["whisper"]["language"] = args.language
    model = WhisperNPU(cfg, device=args.device)

    if args.input.lower() == "mic":
        meter = None if args.no_meter else _make_vu_meter(model.silence_threshold)
        while True:
            print("\nSpeak (silence ends the utterance, or press 'q' to end now)...")
            text = model.listen_once(on_level=meter, stop_key="q")
            if meter is not None:
                _clear_meter_line()
            print(f"→ {text!r}")
            if not args.loop:
                break
    else:
        wav = Path(args.input)
        if not wav.exists():
            raise SystemExit(f"{wav} not found")
        audio = _load_wav(str(wav))
        text, rtf = model.model.transcribe(audio, chunk_length_s=30)
        print("\nTranscription:", text)


if __name__ == "__main__":
    main()
