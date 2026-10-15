# oruk_realtime_asr_python

Streaming speech-to-text for TEN using **Oruk's realtime API**. It streams audio
over an authenticated WebSocket and surfaces interim transcript deltas and a
final transcript per turn, plus optional **phrase-emotion estimates** (vocal
expression descriptors that arrive independently of the transcript).

> Phrase-emotion estimates describe how speech sounds. They are not verified
> assessments of a person's internal state.

For a fully local, key-free option, use `oruk_asr_python` (Orukeet on CPU).

## Audio

Mono **PCM16 at 16 kHz** (`pcm_frame`).

## Credentials

The Oruk API key is read from `params.api_key` / `params.key` (typically wired
from `${env:ORUK_API_KEY}`) and sent only in the WebSocket upgrade's
`Authorization` header — never in the URL.

## Configuration (`property.json` → `params`)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `key` / `api_key` | string | — | Oruk API key (required). |
| `url` | string | `wss://speech-api.oruk.ai/v1/realtime?model=oruk-realtime` | Must be `wss`. |
| `sample_rate` | int | `16000` | Must be 16000. |
| `language` | string | `"auto"` | Oruk locale or `"auto"`. |
| `phrase_emotions` | bool | `true` | Emit phrase-emotion estimates. |
| `diarize` | bool | `false` | Speaker separation. |
| `result_language_default` | string | derived | Locale stamped on results when the server omits one. |

## Turn model

The Oruk realtime socket commits **one turn per connection**: the extension
streams PCM, emits interim results, and on finalize sends a commit to receive
the final transcript, phrase-emotion estimates, and usage before the server
closes. It then reconnects for the next turn. Audio that arrives during a
reconnect is buffered (keep strategy) so the next utterance is not lost.

Phrase-emotion estimates and usage are reported via the vendor-metrics channel;
estimates available before the final transcript are also attached to the final
result's `metadata.asr_info.phrases`.


## Lifecycle review follow-up

The review revision latches invalid configuration, preserves the audio timeline across reconnects, ignores replaced-socket callbacks, bounds recovery to five attempts with backoff, closes per-turn HTTP sessions, and completes finalize on disconnect/timeout. Stop cancels and awaits recovery.

Offline fault-injection coverage is in `oruk_realtime_asr_python/tests/unit/test_lifecycle.py` and covers both extensions. Run `python -m unittest discover -s agents/ten_packages/extension/oruk_realtime_asr_python/tests/unit -v` from `ai_agents/`. The tests stub the TEN boundary and exercise actual extension and transport methods. Native runtime/ASR guarders still need the supported Linux development container and are a merge gate; offline success is not a claim of full live acceptance.
