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
