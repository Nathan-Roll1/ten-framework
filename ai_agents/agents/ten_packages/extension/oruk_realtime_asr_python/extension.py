import asyncio
import copy
import os
from typing import Any, Dict

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
    ModuleErrorVendorInfo,
)
from ten_runtime import (
    AsyncTenEnv,
    AudioFrame,
)

from .config import OrukRealtimeASRConfig
from .const import DUMP_FILE_NAME, MODEL, MODULE_NAME_ASR, SAMPLE_RATE
from .recognition import OrukRealtimeConnection, OrukRealtimeCallback


class OrukRealtimeASRExtension(AsyncASRBaseExtension, OrukRealtimeCallback):
    """Oruk realtime (streaming WebSocket) ASR for TEN.

    Streams PCM16 to Oruk's realtime API, surfacing interim transcript deltas and
    a final transcript per turn plus optional phrase-emotion estimates. The Oruk
    socket commits one turn per connection, so the extension opens a fresh
    connection for each turn and reconnects after the server closes.

    Phrase-emotion estimates describe vocal expression only; they are not
    verified assessments of a person's internal state.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.config: OrukRealtimeASRConfig | None = None
        self.connection: OrukRealtimeConnection | None = None
        self.audio_dumper: Dumper | None = None
        self._reconnect_lock = asyncio.Lock()
        # Per-turn state.
        self._phrases: list[dict[str, Any]] = []
        self._final_sent = False
        self._turn_bytes = 0
        self._emitted_ms = 0

    @override
    def vendor(self) -> str:
        return "oruk"

    @override
    def vendor_metadata(self) -> dict[str, Any]:
        if self.config is None:
            return {}
        fields = {
            "key": self.config.api_key,
            "url": self.config.url,
            "model": MODEL,
        }
        # The base class redacts sensitive fields before reporting.
        return {k: v for k, v in fields.items() if v}

    @override
    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        await super().on_init(ten_env)

        config_json, _ = await ten_env.get_property_to_json("")
        try:
            self.config = OrukRealtimeASRConfig.model_validate_json(config_json)
            self.config.validate_config()
            ten_env.log_info(
                f"config: {self.config.to_json(sensitive_handling=True)}",
                category=LOG_CATEGORY_KEY_POINT,
            )
            if self.config.dump:
                dump_file_path = os.path.join(self.config.dump_path, DUMP_FILE_NAME)
                self.audio_dumper = Dumper(dump_file_path)
                await self.audio_dumper.start()
        except Exception as e:
            ten_env.log_error(f"Invalid Oruk realtime config: {e}")
            self.config = OrukRealtimeASRConfig.model_validate_json("{}")
            await self.send_asr_error(
                ModuleError(
                    module=MODULE_NAME_ASR,
                    code=ModuleErrorCode.FATAL_ERROR.value,
                    message=str(e),
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
        """Open one realtime turn. The base emits CONNECTING before this runs."""
        assert self.config is not None
        if not self.config.api_key:
            error = ModuleError(
                module=MODULE_NAME_ASR,
                code=ModuleErrorCode.FATAL_ERROR.value,
                message="Oruk API key is required but missing or empty",
            )
            await self.send_asr_error(error)
            await self.on_disconnected(code=error.code, message=error.message)
            return

        if self.connection is not None:
            await self.stop_connection()

        self._reset_turn()
        self.connection = OrukRealtimeConnection(
            api_key=self.config.api_key,
            url=self.config.url,
            language=self.config.language,
            phrase_emotions=self.config.phrase_emotions,
            diarize=self.config.diarize,
            sample_rate=self.config.sample_rate,
            callback=self,
            ten_env=self.ten_env,
        )
        try:
            await self.connection.start(request_id=self.uuid)
        except Exception as e:
            self.ten_env.log_error(f"Failed to connect to Oruk realtime: {e}")
            self.connection = None
            vendor_info = ModuleErrorVendorInfo(
                vendor=self.vendor(), code="connect_failed", message=str(e)
            )
            await self.send_asr_error(
                ModuleError(
                    module=MODULE_NAME_ASR,
                    code=ModuleErrorCode.NON_FATAL_ERROR.value,
                    message=str(e),
                ),
                vendor_info,
            )
            await self.on_disconnected(
                code=ModuleErrorCode.NON_FATAL_ERROR.value,
                message=str(e),
                vendor_info=vendor_info,
            )

    @override
    def is_connected(self) -> bool:
        return self.connection is not None and self.connection.is_connected()

    @override
    async def stop_connection(self) -> None:
        if self.connection is not None:
            connection = self.connection
            self.connection = None
            await connection.close()

    @override
    def input_audio_sample_rate(self) -> int:
        assert self.config is not None
        return self.config.sample_rate

    @override
    def buffer_strategy(self) -> ASRBufferConfig:
        # Hold audio while the next turn's socket is (re)connecting.
        return ASRBufferConfigModeKeep(byte_limit=1024 * 1024 * 10)

    @override
    async def send_audio(self, frame: AudioFrame, _session_id: str | None) -> bool:
        if self.connection is None:
            return False
        buf = frame.lock_buf()
        try:
            audio_data = bytes(buf)
            if self.audio_dumper:
                await self.audio_dumper.push_bytes(audio_data)
            await self.connection.send_pcm(audio_data)
            self._turn_bytes += len(audio_data)
            return True
        except Exception as e:
            self.ten_env.log_error(f"Error sending audio to Oruk realtime: {e}")
            return False
        finally:
            frame.unlock_buf(buf)

    @override
    async def finalize(self, _session_id: str | None) -> None:
        """Commit the current turn so the server emits the final transcript."""
        if self.connection is not None:
            await self.connection.commit()

    # ---- Vendor callbacks (OrukRealtimeCallback) ----

    @override
    async def on_open(self) -> None:
        self.ten_env.log_info(
            "Oruk realtime connection opened", category=LOG_CATEGORY_VENDOR
        )
        await self.on_connected()

    @override
    async def on_interim(self, text: str) -> None:
        await self.send_asr_result(
            ASRResult(
                text=text,
                final=False,
                start_ms=self._emitted_ms,
                duration_ms=self._turn_duration_ms(),
                language=self.config.asr_result_language if self.config else "",
                words=[],
                metadata=self._build_metadata(),
            )
        )

    @override
    async def on_final(self, text: str) -> None:
        if self._final_sent:
            return
        self._final_sent = True
        duration_ms = self._turn_duration_ms()
        await self.send_asr_result(
            ASRResult(
                text=text,
                final=True,
                start_ms=self._emitted_ms,
                duration_ms=duration_ms,
                language=self.config.asr_result_language if self.config else "",
                words=[],
                metadata=self._build_metadata(with_phrases=True),
            )
        )
        self._emitted_ms += duration_ms
        await self.send_asr_finalize_end()

    @override
    async def on_phrase_emotion(self, event: dict[str, Any]) -> None:
        self._phrases.append(event)
        # Emotions that arrive after the final transcript are reported as vendor
        # metrics so they are not lost while keeping the text result immutable.
        if self._final_sent:
            await self.send_vendor_metrics({"phrase_emotion": event})

    @override
    async def on_usage(self, usage: dict[str, Any]) -> None:
        await self.send_vendor_metrics({"usage": usage})

    @override
    async def on_error(
        self, message: str, code: str | None, fatal: bool
    ) -> None:
        self.ten_env.log_error(
            f"vendor_error: code={code}, message={message}",
            category=LOG_CATEGORY_VENDOR,
        )
        vendor_info = ModuleErrorVendorInfo(
            vendor=self.vendor(),
            code=str(code) if code else "unknown",
            message=message,
        )
        await self.send_asr_error(
            ModuleError(
                module=MODULE_NAME_ASR,
                code=(
                    ModuleErrorCode.FATAL_ERROR.value
                    if fatal
                    else ModuleErrorCode.NON_FATAL_ERROR.value
                ),
                message=message,
            ),
            vendor_info,
        )

    @override
    async def on_close(self, code: int, message: str) -> None:
        self.ten_env.log_info(
            f"Oruk realtime connection closed: code={code}, message={message}",
            category=LOG_CATEGORY_VENDOR,
        )
        self.connection = None
        await self.on_disconnected(code=0, message="closed")
        # The realtime socket commits one turn per connection; reopen for the
        # next turn unless the extension is shutting down.
        if not self.stopped:
            asyncio.create_task(self._reopen())

    async def _reopen(self) -> None:
        async with self._reconnect_lock:
            if self.stopped or self.is_connected():
                return
            self.ten_env.log_debug("Reopening Oruk realtime connection for next turn")
            await self.start_connection()

    # ---- helpers ----

    def _reset_turn(self) -> None:
        self._phrases = []
        self._final_sent = False
        self._turn_bytes = 0

    def _turn_duration_ms(self) -> int:
        rate = self.config.sample_rate if self.config else SAMPLE_RATE
        return int(self._turn_bytes / 2 / rate * 1000)

    def _build_metadata(self, with_phrases: bool = False) -> Dict[str, Any]:
        base_metadata = copy.deepcopy(self.metadata) if self.metadata else {}
        session_id = base_metadata.pop("session_id", None)
        asr_info = base_metadata
        asr_info["vendor"] = self.vendor()
        asr_info["model"] = MODEL
        if with_phrases and self._phrases:
            asr_info["phrases"] = copy.deepcopy(self._phrases)

        metadata: Dict[str, Any] = {}
        if session_id is not None:
            metadata["session_id"] = session_id
        metadata["asr_info"] = asr_info
        return metadata
