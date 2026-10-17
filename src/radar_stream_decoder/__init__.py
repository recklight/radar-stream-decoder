"""Decoder for the UART output stream of TI mmWave radar sensors.

The Type-Length-Value packets that TI mmWave sensors (IWR14xx/16xx/18xx/68xx)
send over their data port are turned into NumPy data with physical units.
The main pieces:

* :class:`FrameAssembler`: incremental packet assembly that resynchronises
  after junk or lost bytes.
* :func:`decode_packet` / :class:`RadarFrame`: decoding of one packet against
  a schema (out-of-box demo SDK 1.x/2.x and 3.x, Vital Signs lab, the 3D and
  2D people counting labs, Radar Toolbox people tracking), with the header
  layout and checksum checked.
* :mod:`~radar_stream_decoder.tracking`: tracker targets, presence and the
  per-point target IDs (:func:`link_target_associations`).
* :func:`parse_cli_config` / :class:`SensorConfig`: FMCW parameters from a
  ``.cfg`` file (range and Doppler resolution, antenna counts ...) under a
  selectable :class:`FmcwConvention`.
* :func:`analyze_capture` / :class:`CaptureReport`: capture forensics
  (repeats, drops, restarts, timestamp unwrapping).
* :class:`SyntheticScene`, :func:`generate_packets`, :func:`inject_faults`:
  synthetic streams for tests and demonstrations.
* :mod:`~radar_stream_decoder.export`, :mod:`~radar_stream_decoder.sources`,
  :mod:`~radar_stream_decoder.serial_link`: output files, byte sources and
  live capture. :mod:`~radar_stream_decoder.port_emulator` stands in for the
  sensor's two serial ports.

The PySide6 viewer in :mod:`radar_stream_decoder.viz` is not imported here,
so decoding works without the ``gui`` extra installed.
"""

from __future__ import annotations

from .angle_map import AzimuthHeatmap, angle_axis_deg, angle_spectrum, polar_to_cartesian
from .assembler import AssemblerStats, FrameAssembler
from .capture import (
    CaptureAnalysis,
    CaptureReport,
    analyze_capture,
    deduplicate,
    iter_capture,
    unwrap_cpu_cycles,
)
from .config import DecoderConfig, NoiseEstimator, ViewerConfig
from .constants import MAGIC_WORD, Platform
from .encoder import HeaderTemplate, encode_frame
from .errors import (
    ConfigError,
    DecodeError,
    DecodeIssue,
    IssueCode,
    SchemaError,
    Severity,
    UnsupportedFormatError,
)
from .frame import RadarFrame, decode_packet
from .header import FrameHeader, HeaderLayout, SdkVersion
from .payloads import (
    MagnitudeProfile,
    PayloadKind,
    PointCloud,
    PointSideInfo,
    ProcessingStats,
    RangeDopplerMap,
    SnrMethod,
    TemperatureReport,
    UnknownTlv,
)
from .schema import PacketSchema, detect_schema
from .sensor_config import (
    FmcwConvention,
    SensorConfig,
    WavelengthReference,
    load_cli_config,
    parse_cli_config,
)
from .sources import FileByteSource, FrameStream, SerialByteSource, SyntheticByteSource
from .synthetic import FaultPlan, SyntheticScene, Target, generate_packets, inject_faults
from .tracking import (
    PresenceIndication,
    TargetAssociation,
    TargetList,
    link_target_associations,
)
from .units import Log2Combining
from .vital_signs import RangeSegment, VitalSignsSummary, phase_displacement_mm

__version__ = "0.1.0"

__all__ = [
    "MAGIC_WORD",
    "AssemblerStats",
    "AzimuthHeatmap",
    "CaptureAnalysis",
    "CaptureReport",
    "ConfigError",
    "DecodeError",
    "DecodeIssue",
    "DecoderConfig",
    "FaultPlan",
    "FileByteSource",
    "FmcwConvention",
    "FrameAssembler",
    "FrameHeader",
    "FrameStream",
    "HeaderLayout",
    "HeaderTemplate",
    "IssueCode",
    "Log2Combining",
    "MagnitudeProfile",
    "NoiseEstimator",
    "PacketSchema",
    "PayloadKind",
    "Platform",
    "PointCloud",
    "PointSideInfo",
    "PresenceIndication",
    "ProcessingStats",
    "RadarFrame",
    "RangeDopplerMap",
    "RangeSegment",
    "SchemaError",
    "SdkVersion",
    "SensorConfig",
    "SerialByteSource",
    "Severity",
    "SnrMethod",
    "SyntheticByteSource",
    "SyntheticScene",
    "Target",
    "TargetAssociation",
    "TargetList",
    "TemperatureReport",
    "UnknownTlv",
    "UnsupportedFormatError",
    "ViewerConfig",
    "VitalSignsSummary",
    "WavelengthReference",
    "__version__",
    "analyze_capture",
    "angle_axis_deg",
    "angle_spectrum",
    "decode_packet",
    "deduplicate",
    "detect_schema",
    "encode_frame",
    "generate_packets",
    "inject_faults",
    "iter_capture",
    "link_target_associations",
    "load_cli_config",
    "parse_cli_config",
    "phase_displacement_mm",
    "polar_to_cartesian",
    "unwrap_cpu_cycles",
]
