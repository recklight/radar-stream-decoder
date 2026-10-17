"""Incremental packet assembly with resynchronisation.

A UART stream arrives in arbitrary pieces, may start in the middle of a
packet, may contain bytes that are not packets at all, and loses bytes when a
receive buffer overflows. :class:`FrameAssembler` takes the stream in any
chunking (a whole file at once and one byte at a time give the same frames)
and picks up again after each of these faults.

State machine
-------------
1. Hunt: search for the sync word. Bytes before it are skipped; the last
   seven bytes are kept in case a sync word straddles two chunks.
2. Validate: once enough bytes are buffered (36 for the padded layout,
   48 or 52 for the checksummed people counting layouts), identify the header
   layout and check it (version, platform family, TLV count, a length that
   fits the TLVs and the size limit, and the checksum where the layout has
   one). A failing candidate costs exactly one byte: the search resumes at the
   next byte, never at the next chunk.
3. Watch for cuts: while the declared length is not yet complete, look for
   another sync word inside it. If one starts a plausible header, the current
   packet was cut short on the link. In lenient mode its complete TLVs are
   still decoded (a frame flagged ``truncated``); in strict mode it is
   rejected. Either way decoding continues at the inner sync word, so the
   packet that follows is not lost. A complete span whose last bytes could be
   the start of a sync word waits for up to seven more bytes (or the end of the
   stream) first, so a sync word straddling the declared end is found however
   the stream was chunked.
4. Decode the complete packet. A packet that does not decode is
   rejected (one byte skipped, hunt again); a packet whose every TLV failed
   is treated the same way. A packet that is intact but belongs to no known
   firmware family is rejected with an ``unsupported-format`` issue.

At the end of the stream :meth:`FrameAssembler.finish` decodes what can still
be decoded and records an unfinished packet as a truncated tail instead of
raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import DecoderConfig
from .constants import MAGIC_WORD
from .errors import DecodeError, DecodeIssue, IssueCode, Severity, UnsupportedFormatError
from .frame import RadarFrame, decode_packet
from .header import HeaderFields, HeaderVerdict, examine_header
from .sensor_config import SensorConfig

_LOGGER = logging.getLogger(__name__)

_MAGIC_LEN = len(MAGIC_WORD)


@dataclass
class AssemblerStats:
    """Counters of an assembler's work.

    Attributes
    ----------
    bytes_in : int
        Bytes fed.
    bytes_skipped : int
        Bytes that belonged to no decoded frame (junk, rejected candidates,
        the parts of cut packets that could not be salvaged).
    frames_decoded : int
        Frames returned, salvaged ones included.
    frames_rejected : int
        Sync-word candidates or packets that were discarded.
    frames_salvaged : int
        Frames decoded from packets that were cut short.
    resyncs : int
        Times synchronisation was lost after a frame had been decoded.
    truncated_tail_bytes : int
        Bytes of an unfinished packet at the end of the stream.
    issues : list of DecodeIssue
        Stream-level issues (bounded; see ``issue_count`` for the total).
    issue_count : int
        Stream-level issues raised, recorded or not.
    """

    bytes_in: int = 0
    bytes_skipped: int = 0
    frames_decoded: int = 0
    frames_rejected: int = 0
    frames_salvaged: int = 0
    resyncs: int = 0
    truncated_tail_bytes: int = 0
    issues: list[DecodeIssue] = field(default_factory=list)
    issue_count: int = 0

    def as_dict(self) -> dict[str, int]:
        """Counters by name (issues excluded)."""
        return {
            "bytes_in": self.bytes_in,
            "bytes_skipped": self.bytes_skipped,
            "frames_decoded": self.frames_decoded,
            "frames_rejected": self.frames_rejected,
            "frames_salvaged": self.frames_salvaged,
            "resyncs": self.resyncs,
            "truncated_tail_bytes": self.truncated_tail_bytes,
            "issue_count": self.issue_count,
        }


class FrameAssembler:
    """Turn a byte stream, fed in arbitrary chunks, into decoded frames.

    Parameters
    ----------
    config : DecoderConfig, optional
        Decoder settings; ``strict`` and ``max_packet_bytes`` matter here.
    sensor : SensorConfig, optional
        Radar configuration passed on to the decoder.
    max_recorded_issues : int, optional
        Stream-level issues kept in :attr:`AssemblerStats.issues`.

    Examples
    --------
    >>> from radar_stream_decoder.synthetic import SyntheticScene, generate_packets
    >>> stream = b"noise" + b"".join(generate_packets(SyntheticScene(seed=1), 3))
    >>> assembler = FrameAssembler()
    >>> frames = [f for i in range(0, len(stream), 7) for f in assembler.feed(stream[i:i + 7])]
    >>> [f.frame_number for f in frames], assembler.stats.bytes_skipped
    ([1, 2, 3], 5)
    """

    def __init__(
        self,
        config: DecoderConfig | None = None,
        sensor: SensorConfig | None = None,
        *,
        max_recorded_issues: int = 1000,
    ) -> None:
        self.config = config or DecoderConfig()
        self.sensor = sensor
        self.max_recorded_issues = max_recorded_issues
        self._buffer = bytearray()
        self._stats = AssemblerStats()
        self._consumed = 0
        self._locked = False
        self._candidate: HeaderFields | None = None
        self._scan_from = 1
        self._final = False
        self.reset()

    def reset(self) -> None:
        """Forget all buffered bytes and counters, ready for a new stream."""
        self._buffer = bytearray()
        self._stats = AssemblerStats()
        self._consumed = 0
        self._locked = False
        self._candidate = None
        self._scan_from = 1
        self._final = False

    # -- public interface --------------------------------------------------------

    @property
    def stats(self) -> AssemblerStats:
        """Live counters (the same object across calls)."""
        return self._stats

    @property
    def buffered_bytes(self) -> int:
        """Bytes waiting for the rest of their packet."""
        return len(self._buffer)

    def feed(self, data: bytes | bytearray | memoryview) -> list[RadarFrame]:
        """Add bytes and return every frame they complete.

        Raises
        ------
        RuntimeError
            After :meth:`finish`, until :meth:`reset` is called.
        """
        if self._final:
            raise RuntimeError("the stream is finished; call reset() to start a new one")
        self._buffer += data
        self._stats.bytes_in += len(data)
        return self._drain()

    def finish(self) -> list[RadarFrame]:
        """End the stream: decode what can still be decoded, account for the rest.

        A packet still waiting for bytes, including one that ends inside its
        header, is recorded as a truncated tail (``truncated_tail_bytes``); in
        lenient mode its complete TLVs are decoded first. No exception is
        raised for an unfinished stream.
        """
        if self._final:
            return []
        self._final = True
        frames = self._drain()
        if self._buffer:
            if self._candidate is not None or self._buffer.startswith(MAGIC_WORD):
                frames.extend(self._close_tail())
            else:
                leftover = len(self._buffer)
                self._skip(leftover, "trailing bytes without a sync word")
        return frames

    def flush(self) -> AssemblerStats:
        """End the stream (see :meth:`finish`) and return the final counters.

        Frames that :meth:`finish` would still produce are discarded; call
        :meth:`finish` first when they matter.
        """
        self.finish()
        return self._stats

    # -- bookkeeping -------------------------------------------------------------

    def _record(self, issue: DecodeIssue) -> None:
        self._stats.issue_count += 1
        if len(self._stats.issues) < self.max_recorded_issues:
            self._stats.issues.append(issue)
        _LOGGER.debug("%s", issue)

    def _skip(self, count: int, reason: str) -> None:
        if count <= 0:
            return
        if self._locked:
            self._stats.resyncs += 1
            self._locked = False
            self._record(
                DecodeIssue(
                    IssueCode.SKIPPED_BYTES,
                    f"lost synchronisation: {reason}",
                    Severity.WARNING,
                    offset=self._consumed,
                )
            )
        del self._buffer[:count]
        self._consumed += count
        self._stats.bytes_skipped += count

    def _consume(self, count: int) -> None:
        del self._buffer[:count]
        self._consumed += count
        self._locked = True
        self._candidate = None
        self._scan_from = 1

    def _reject(
        self,
        message: str,
        skip: int = 1,
        frame: int | None = None,
        code: IssueCode = IssueCode.REJECTED_PACKET,
    ) -> None:
        self._stats.frames_rejected += 1
        self._record(DecodeIssue(code, message, Severity.WARNING, frame, self._consumed))
        self._candidate = None
        self._scan_from = 1
        self._skip(skip, message)

    def _examine(self, offset: int) -> HeaderVerdict:
        schema = self.config.schema
        return examine_header(
            self._buffer,
            offset,
            max_packet_bytes=self.config.max_packet_bytes,
            max_tlvs=self.config.max_tlvs,
            strict=self.config.strict,
            layout=schema.header_layout if schema is not None else None,
        )

    def _accept(self, frame: RadarFrame, length: int) -> RadarFrame:
        self._consume(length)
        self._stats.frames_decoded += 1
        if frame.truncated:
            self._stats.frames_salvaged += 1
        return frame

    # -- the state machine -------------------------------------------------------------

    def _drain(self) -> list[RadarFrame]:
        frames: list[RadarFrame] = []
        while True:
            outcome = self._step()
            if outcome is None:
                return frames
            if isinstance(outcome, RadarFrame):
                frames.append(outcome)

    def _step(self) -> RadarFrame | bool | None:
        """Advance once: a frame, ``True`` for progress without a frame, ``None`` to wait."""
        buffer = self._buffer
        if self._candidate is None:
            start = buffer.find(MAGIC_WORD)
            if start < 0:
                self._skip(len(buffer) - (_MAGIC_LEN - 1), "no sync word")
                return None
            if start > 0:
                self._skip(start, f"{start} bytes before the sync word")
            verdict = self._examine(0)
            if verdict.pending:
                return None
            if verdict.fields is None:
                self._reject(
                    f"sync word with an implausible header: {verdict.problem}", code=verdict.code
                )
                return True
            self._candidate = verdict.fields
            self._scan_from = 1

        fields = self._candidate
        length = fields.packet_length
        cut = self._find_inner_header(length)
        if cut is None:
            return None
        if cut > 0:
            return self._close_cut_packet(fields, cut)
        if len(buffer) < length:
            return None
        try:
            frame = decode_packet(bytes(buffer[:length]), self.config, self.sensor)
        except DecodeError as exc:
            code = (
                IssueCode.UNSUPPORTED_FORMAT
                if isinstance(exc, UnsupportedFormatError)
                else IssueCode.REJECTED_PACKET
            )
            self._reject(f"packet rejected: {exc}", frame=fields.frame_number, code=code)
            return True
        if frame.has_errors and not frame.payloads:
            self._reject(
                f"no TLV of the packet could be decoded: {frame.issues[0]}",
                frame=fields.frame_number,
            )
            return True
        return self._accept(frame, length)

    def _find_inner_header(self, length: int) -> int | None:
        """Position of a plausible header inside the declared span, 0 for none, None to wait."""
        buffer = self._buffer
        end = min(len(buffer), length + _MAGIC_LEN - 1)
        while True:
            inner = buffer.find(MAGIC_WORD, self._scan_from, end)
            if inner < 0:
                self._scan_from = max(self._scan_from, end - _MAGIC_LEN + 1)
                if not self._final and self._sync_word_may_straddle_end(length):
                    return None
                return 0
            verdict = self._examine(inner)
            if verdict.pending:
                if self._final:
                    self._scan_from = inner + 1
                    continue
                return None
            if verdict.fields is not None:
                return inner
            self._scan_from = inner + 1

    def _sync_word_may_straddle_end(self, length: int) -> bool:
        """Whether the declared span is complete but may end inside the next sync word.

        A packet that lost a few bytes on the link is followed by the next
        packet's sync word *inside* its declared span, possibly in its last
        seven bytes. Those are only recognisable once the bytes after the span
        have arrived, so a complete span whose tail is a prefix of the sync word
        waits for them (or for the end of the stream). Without this, the result
        would depend on where the chunks happen to be split.
        """
        buffer = self._buffer
        have = len(buffer)
        if not length <= have < length + _MAGIC_LEN - 1:
            return False
        first = max(self._scan_from, have - _MAGIC_LEN + 1)
        return any(MAGIC_WORD.startswith(buffer[start:have]) for start in range(first, length))

    def _close_cut_packet(self, fields: HeaderFields, cut: int) -> RadarFrame | bool:
        """The packet ends early at ``cut``, where the next packet starts."""
        message = (
            f"frame {fields.frame_number} cut short: {cut} of {fields.packet_length} bytes "
            "before the next sync word"
        )
        if not self.config.strict:
            try:
                frame = decode_packet(
                    bytes(self._buffer[:cut]), self.config, self.sensor, available=cut
                )
            except DecodeError:
                frame = None
            if frame is not None and frame.payloads:
                return self._accept(frame, cut)
        self._reject(message, skip=cut, frame=fields.frame_number)
        return True

    def _close_tail(self) -> list[RadarFrame]:
        """Account for (and in lenient mode salvage) an unfinished final packet."""
        fields = self._candidate
        have = len(self._buffer)
        if fields is None:
            # Only the sync word and part of the header arrived.
            message = f"stream ended {have} bytes into a packet, inside its header"
            frame_number = None
        else:
            message = (
                f"stream ended {fields.packet_length - have} bytes before the end of frame "
                f"{fields.frame_number}"
            )
            frame_number = fields.frame_number
        self._record(
            DecodeIssue(
                IssueCode.TRUNCATED_TAIL, message, Severity.WARNING, frame_number, self._consumed
            )
        )
        self._stats.truncated_tail_bytes += have
        frames: list[RadarFrame] = []
        if fields is not None and not self.config.strict:
            try:
                frame = decode_packet(bytes(self._buffer), self.config, self.sensor, available=have)
            except DecodeError:
                frame = None
            if frame is not None and frame.payloads:
                # The frame already carries its own truncation issue; the
                # stream-level tail issue stays with the assembler counters.
                self._stats.frames_decoded += 1
                self._stats.frames_salvaged += 1
                frames.append(frame)
        del self._buffer[:]
        self._consumed += have
        self._candidate = None
        return frames
