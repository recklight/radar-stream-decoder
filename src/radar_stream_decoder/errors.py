"""Exceptions and the structured record of a decoding problem.

Exceptions are kept for what the caller has to handle: a buffer that is not
a packet at all (:class:`DecodeError`), a packet of a firmware family the
decoder does not know (:class:`UnsupportedFormatError`), a radar configuration
that cannot be parsed (:class:`ConfigError`), an impossible schema request
(:class:`SchemaError`).

Everything that can be survived becomes a :class:`DecodeIssue` instead: an
unknown TLV type, a payload whose length disagrees with its contents, a packet
cut short on the link. Issues travel with the decoded frame, so one bad packet
never stops a capture from being read. In strict mode an error-level issue
turns into a :class:`DecodeError`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    """How much an issue undermines the decoded data."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueCode(str, Enum):
    """Stable identifiers of the problems the decoder reports."""

    INVALID_HEADER = "invalid-header"
    HEADER_LENGTH_CONFLICT = "header-length-conflict"
    TLV_OVERFLOW = "tlv-overflow"
    TRUNCATED_PACKET = "truncated-packet"
    SHORTENED_PACKET = "shortened-packet"
    EXCESS_PADDING = "excess-padding"
    CORRUPT_PADDING = "corrupt-padding"
    UNEVEN_PADDING = "uneven-padding"
    HEADER_CHECKSUM_MISMATCH = "checksum-mismatch"
    UNSUPPORTED_FORMAT = "unsupported-format"
    UNKNOWN_TLV = "unknown-tlv"
    DUPLICATE_TLV = "duplicate-tlv"
    LENGTH_MISMATCH = "length-mismatch"
    INVALID_VALUE = "invalid-value"
    POINT_COUNT_MISMATCH = "point-count-mismatch"
    MISSING_DOPPLER_BINS = "missing-doppler-bins"
    UNRESOLVED_SHAPE = "unresolved-shape"
    MISSING_RANGE_START = "missing-range-start"
    SIDE_INFO_MISMATCH = "side-info-mismatch"
    AMBIGUOUS_RECORD_SIZE = "ambiguous-record-size"
    ASSOCIATION_MISMATCH = "association-mismatch"
    SKIPPED_BYTES = "skipped-bytes"
    REJECTED_PACKET = "rejected-packet"
    TRUNCATED_TAIL = "truncated-tail"
    FRAME_CONFLICT = "frame-conflict"
    FRAME_REPLACED = "frame-replaced"
    FRAME_RESET = "frame-reset"


@dataclass(frozen=True)
class DecodeIssue:
    """A problem found while decoding, kept next to the data it concerns.

    Attributes
    ----------
    code : IssueCode
        Machine-readable category.
    message : str
        Human-readable explanation, specific enough to act on.
    severity : Severity
        ``ERROR`` means part of the frame could not be trusted or decoded.
    frame : int, optional
        Frame number of the packet, when the header was readable.
    offset : int, optional
        Byte offset of the problem, within the packet for payload issues and
        within the stream for assembler issues.

    Examples
    --------
    >>> issue = DecodeIssue(IssueCode.UNKNOWN_TLV, "TLV type 42 is not defined", frame=7)
    >>> str(issue)
    'warning unknown-tlv (frame 7): TLV type 42 is not defined'
    """

    code: IssueCode
    message: str
    severity: Severity = Severity.WARNING
    frame: int | None = None
    offset: int | None = None

    def __str__(self) -> str:
        where = []
        if self.frame is not None:
            where.append(f"frame {self.frame}")
        if self.offset is not None:
            where.append(f"offset {self.offset}")
        location = f" ({', '.join(where)})" if where else ""
        return f"{self.severity.value} {self.code.value}{location}: {self.message}"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary."""
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "frame": self.frame,
            "offset": self.offset,
        }


class DecodeError(ValueError):
    """A buffer could not be decoded as a packet.

    Parameters
    ----------
    message : str
        What went wrong.
    issues : iterable of DecodeIssue, optional
        The issues collected before giving up (strict mode).
    """

    def __init__(self, message: str, issues: Iterable[DecodeIssue] = ()) -> None:
        super().__init__(message)
        self.issues: tuple[DecodeIssue, ...] = tuple(issues)


class SchemaError(DecodeError):
    """A packet schema was requested that does not exist or does not fit the data."""


class UnsupportedFormatError(DecodeError):
    """A packet is intact but no known firmware family explains it.

    The decoder does not force such a packet into the nearest schema; its
    ``issues`` carry one ``unsupported-format`` issue saying what was found.
    """


class ConfigError(ValueError):
    """A radar CLI configuration or a decoder setting is invalid.

    Parameters
    ----------
    message : str
        What is wrong.
    line : int, optional
        1-based line number in the configuration text, when known.
    """

    def __init__(self, message: str, line: int | None = None) -> None:
        self.line = line
        super().__init__(f"line {line}: {message}" if line is not None else message)
