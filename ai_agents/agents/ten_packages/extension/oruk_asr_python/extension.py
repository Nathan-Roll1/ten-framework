import asyncio
import copy
import os
from typing import Any, Dict

import numpy as np
from typing_extensions import override

from ten_ai_base.asr import (
    ASRBufferConfig,
    ASRBufferConfigModeKeep,
    ASRResult,
    AsyncASRBaseExtension,
)
from ten_ai_base.const import (
    LOG_CATEGORY_KEY_POINT,
    LOG_CATEGORY_VENDOR,
)
from ten_ai_base.dumper import Dumper
from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
)
from ten_runtime import (
    AsyncTenEnv,
    AudioFrame,
)

from .config import OrukASRConfig
from .const import DUMP_FILE_NAME, MODULE_NAME_ASR, ORUKEET_MODEL
from .recognizer import OrukeetRecognizer


class OrukASRExtension(AsyncASRBaseExtension):
    """Local Orukeet ASR for TEN.

    Orukeet (NVIDIA Parakeet TDT 0.6B v3 + Gabor kernels) runs on CPU via
    onnx-asr. There is no network connection: "connecting" means loading the
    INT8 ONNX model once. Audio is buffered per utterance and transcribed when
    the framework finalizes the turn (typically on a VAD stop / asr_finalize),
    so this extension emits final results only, with automatic language
    detection. It needs no API key and never uploads audio.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.config: OrukASRConfig | None = None
        self.recognizer: OrukeetRecognizer | None = None
        self.audio_dumper: Dumper | None = None
        self._connected = False
        self._init_failed = False
        # Current utterance PCM16 bytes and the stream offset where it began.
        self._utterance = bytearray()
        self._utterance_start_ms = 0

    @override
    def vendor(self) -> str:
        return "oruk"

    @override
    def vendor_metadata(self) -> dict[str, Any]:
        # No credentials or URLs: the local model is fixed to Orukeet.
        return {"model": ORUKEET_MODEL}

    @override
    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        await super().on_init(ten_env)

        config_json, _ = await ten_env.get_property_to_json("")
        try:
            self.config = OrukASRConfig.model_validate_json(config_json)
            self.config.validate_config()
            ten_env.log_info(
                f"config: {self.config.to_json()}",
                category=LOG_CATEGORY_KEY_POINT,
            )
            if self.config.dump:
                dump_file_path = os.path.join(
                    self.config.dump_path, DUMP_FILE_NAME
                )
                self.audio_dumper = Dumper(dump_file_path)
                await self.audio_dumper.start()
        except Exception:
            ten_env.log_error("Invalid Oruk ASR configuration")
            self._init_failed = True
            self.config = OrukASRConfig.model_validate_json("{}")
            await self.send_asr_error(
                ModuleError(
                    module=MODULE_NAME_ASR,
                    code=ModuleErrorCode.FATAL_ERROR.value,
                    message="Invalid Oruk ASR configuration",
                ),
            )

    @override
    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        await super().on_deinit(ten_env)
        if self.audio_dumper:
            await self.audio_dumper.stop()
            self.audio_dumper = None

    @override
    async def start_connection(self) -> None:
        """Load the local model once. The base emits CONNECTING before this runs."""
        if self._init_failed or self.stopped:
            return
        assert self.config is not None
        self.ten_env.log_info(
            "Loading Orukeet model (CPU, INT8 ONNX)",
            category=LOG_CATEGORY_VENDOR,
        )
        began = asyncio.get_running_loop().time()
        try:
            if self.recognizer is None:
                self.recognizer = OrukeetRecognizer(
                    cache_dir=self.config.cache_dir,
                    local_files_only=self.config.local_files_only,
                )
            await asyncio.to_thread(self.recognizer.prewarm)
            self._connected = True
            self._utterance.clear()
            self._utterance_start_ms = int(
                self.audio_timeline.get_total_user_audio_duration()
            )
            await self.on_connected()
            await self.send_connect_delay_metrics(
                int((asyncio.get_running_loop().time() - began) * 1000)
            )
            self.ten_env.log_info(
                "Orukeet model ready", category=LOG_CATEGORY_VENDOR
            )
        except Exception as e:
            self.ten_env.log_error(f"Failed to load Orukeet model: {e}")
            error = ModuleError(
                module=MODULE_NAME_ASR,
                code=ModuleErrorCode.FATAL_ERROR.value,
                message=str(e),
            )
            await self.send_asr_error(error)
            await self.on_disconnected(code=error.code, message=error.message)

    @override
    def is_connected(self) -> bool:
        return (
            self._connected
            and self.recognizer is not None
            and self.recognizer.loaded
        )

    @override
    async def stop_connection(self) -> None:
        self._connected = False
        self._utterance.clear()
        if self.recognizer is not None:
            recognizer = self.recognizer
            self.recognizer = None
            await asyncio.to_thread(recognizer.close)

    @override
    def input_audio_sample_rate(self) -> int:
        assert self.config is not None
        return self.config.sample_rate

    @override
    def buffer_strategy(self) -> ASRBufferConfig:
        # Keep audio that arrives before the model finishes loading.
        return ASRBufferConfigModeKeep(byte_limit=1024 * 1024 * 10)

    @override
    async def send_audio(
        self, frame: AudioFrame, _session_id: str | None
    ) -> bool:
        buf = frame.lock_buf()
        try:
            audio_data = bytes(buf)
            if self.audio_dumper:
                await self.audio_dumper.push_bytes(audio_data)
            if not self._utterance:
                self._utterance_start_ms = int(
                    self.audio_timeline.get_total_user_audio_duration()
                )
            self._utterance.extend(audio_data)
            self.audio_timeline.add_user_audio(
                len(audio_data) / 2 / self.config.sample_rate * 1000
            )
            return True
        except Exception as e:
            self.ten_env.log_error(f"Error buffering audio for Orukeet: {e}")
            return False
        finally:
            frame.unlock_buf(buf)

    @override
    async def finalize(self, _session_id: str | None) -> None:
        """Transcribe the buffered utterance and emit a single final result."""
        assert self.config is not None

        pcm = bytes(self._utterance)
        # Drop a trailing odd byte so int16 framing stays aligned.
        if len(pcm) % 2:
            pcm = pcm[:-1]
        self._utterance.clear()

        sample_rate = self.config.sample_rate
        duration_ms = int(len(pcm) / 2 / sample_rate * 1000)
        start_ms = self._utterance_start_ms
        self._utterance_start_ms += duration_ms

        if pcm and self.recognizer is not None:
            try:
                samples = (
                    np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                text = await asyncio.to_thread(
                    self.recognizer.recognize, samples, sample_rate
                )
            except Exception as e:
                self.ten_env.log_error(f"Orukeet recognition failed: {e}")
                await self.send_asr_error(
                    ModuleError(
                        module=MODULE_NAME_ASR,
                        code=ModuleErrorCode.NON_FATAL_ERROR.value,
                        message=str(e),
                    ),
                )
                text = ""

            if text:
                await self.send_asr_result(
                    ASRResult(
                        text=text,
                        final=True,
                        start_ms=start_ms,
                        duration_ms=duration_ms,
                        language=self.config.asr_result_language,
                        words=[],
                        metadata=self._build_metadata(),
                    )
                )

        await self.send_asr_finalize_end()

    def _build_metadata(self) -> Dict[str, Any]:
        """session_id at the root; vendor details nested under asr_info."""
        base_metadata = copy.deepcopy(self.metadata) if self.metadata else {}
        session_id = base_metadata.pop("session_id", None)
        asr_info = base_metadata
        asr_info["vendor"] = self.vendor()
        asr_info["model"] = ORUKEET_MODEL

        metadata: Dict[str, Any] = {}
        if session_id is not None:
            metadata["session_id"] = session_id
        metadata["asr_info"] = asr_info
        return metadata
