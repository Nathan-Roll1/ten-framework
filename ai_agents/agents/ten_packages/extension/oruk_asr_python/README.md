# oruk_asr_python

Local, on-device speech-to-text for TEN using **Orukeet** — Oruk's multilingual
ASR model (NVIDIA Parakeet TDT 0.6B v3 with fitted Gabor kernels, 25 languages
with automatic detection). It runs entirely on CPU through
[`onnx-asr`](https://pypi.org/project/onnx-asr/) with the Orukeet INT8 ONNX
export, so it needs **no API key and uploads no audio**.

This extension transcribes complete utterances. The framework's turn detection
(VAD / `asr_finalize`) marks the end of a turn; the buffered audio is then
transcribed and a single **final** result is emitted. It does not produce
interim text, word timestamps, or speaker labels. For streaming interim results
and phrase-emotion estimates, use `oruk_realtime_asr_python`.

## Audio

Mono **PCM16 at 16 kHz** (`pcm_frame`).

## Configuration (`property.json` → `params`)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `sample_rate` | int | `16000` | Must be 16000. |
| `language` | string | `"auto"` | Orukeet auto-detects; must be `"auto"`. |
| `result_language_default` | string | `""` | Optional locale stamped on every result (e.g. `"en-US"`). |
| `cache_dir` | string | Hugging Face default | Where the model is cached. |
| `local_files_only` | bool | `false` | Require a cached model instead of downloading. |
| `model` | string | `"orukeet"` | Fixed to Orukeet. |

## Model download

On first connect, the pinned Orukeet INT8 ONNX export (~486 MB) is downloaded
from Hugging Face (`oruk/orukeet`) and verified by checksum, then loaded once.
Set `cache_dir` to control the cache location, or `local_files_only: true` to
require a pre-provisioned cache in offline deployments.

## Licensing

Orukeet code is MIT; the model weights and Gabor kernels are CC BY-SA 4.0. See
the model card at https://huggingface.co/oruk/orukeet.


## Lifecycle review follow-up

The review revision latches invalid configuration, preserves the audio timeline across reconnects, ignores replaced-socket callbacks, bounds recovery to five attempts with backoff, closes per-turn HTTP sessions, and completes finalize on disconnect/timeout. Stop cancels and awaits recovery.

Offline fault-injection coverage is in `oruk_realtime_asr_python/tests/unit/test_lifecycle.py` and covers both extensions. Run `python -m unittest discover -s agents/ten_packages/extension/oruk_realtime_asr_python/tests/unit -v` from `ai_agents/`. The tests stub the TEN boundary and exercise actual extension and transport methods. Native runtime/ASR guarders still need the supported Linux development container and are a merge gate; offline success is not a claim of full live acceptance.

The local model's 25-language set includes Spanish and does not include Chinese. The native multilingual guarder therefore exercises English and Spanish using the framework's existing supported-language selection. A caller-supplied result locale is metadata, not a detected-language or accuracy result.
