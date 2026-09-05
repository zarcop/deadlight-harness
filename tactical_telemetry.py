"""Edge naval C2 tactical telemetry simulation and sliding-window state buffer.

Self-contained module intended for disconnected / low-bandwidth edge nodes:

* :class:`TacticalCommandEvent`  -- validated wire schema for one telemetry frame.
* :class:`MockTelemetryGenerator` -- deterministic, scenario-driven frame source.
* :class:`TacticalStateWindow`    -- fixed-size sliding buffer that renders its
  contents as a compact, transition-focused natural-language summary suitable
  for semantic vector embedding (anomaly retrieval over recent unit state).

Run ``python tactical_telemetry.py`` for a demonstration of every scenario.
"""

from __future__ import annotations

import math
import random
import textwrap
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Deque, Iterable, Iterator, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "EmconState",
    "Scenario",
    "TacticalCommandEvent",
    "MockTelemetryGenerator",
    "TacticalStateWindow",
]

# --------------------------------------------------------------------------- #
# Doctrinal thresholds (single source of truth for both simulation and text)
# --------------------------------------------------------------------------- #

CORRIDOR_WARNING_M: float = 200.0
"""Deviation from the pre-planned tactical route that warrants attention."""

CORRIDOR_HAZARD_M: float = 800.0
"""Deviation beyond which the unit is considered inside the hazard zone."""

PATROL_SPEED_CEILING_KTS: float = 25.0
"""Speed above which the unit has left the nominal patrol envelope."""

RADAR_BREACH_KW: float = 25.0
"""Surface-search radar output observed during an emissions-control breach."""


class EmconState(str, Enum):
    """Emissions control posture declared by the unit."""

    ALPHA_SILENT = "ALPHA_SILENT"
    BRAVO_RESTRICTED = "BRAVO_RESTRICTED"
    CHARLIE_OPEN = "CHARLIE_OPEN"


class Scenario(str, Enum):
    """Operational scenarios the mock generator can synthesize."""

    NOMINAL = "NOMINAL"
    EMCON_BREACH = "EMCON_BREACH"
    NAV_DIVERGENCE = "NAV_DIVERGENCE"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TacticalCommandEvent(BaseModel):
    """A single telemetry frame emitted by a tactical unit.

    Note that EMCON consistency (``radar_rf_kw == 0.0`` and ``ais_active is
    False`` while EMCON ALPHA is declared) is deliberately *not* enforced as a
    validation rule: an inconsistent frame is exactly the signal a C2 node needs
    to surface. It is exposed through :attr:`is_emcon_violation` instead.
    """

    model_config = ConfigDict(frozen=True, use_enum_values=False)

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the fix.")
    unit_id: str = Field(..., min_length=1, description='Unit callsign, e.g. "USV-GHOST-01".')
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    speed_kts: float = Field(..., ge=0.0, description="Speed over ground in knots.")
    course_deg: float = Field(..., ge=0.0, lt=360.0, description="Course over ground, true.")
    emcon_state: EmconState = Field(..., description="Declared emissions control posture.")
    radar_rf_kw: float = Field(..., ge=0.0, description="Radiated surface-search radar power.")
    ais_active: bool = Field(..., description="AIS transponder transmitting.")
    corridor_deviation_m: float = Field(
        ..., ge=0.0, description="Distance from the pre-planned tactical route, in metres."
    )

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        """Reject anything that is not parseable ISO-8601."""
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @property
    def is_emcon_violation(self) -> bool:
        """True when the unit is emitting while EMCON ALPHA is declared."""
        return self.emcon_state is EmconState.ALPHA_SILENT and (
            self.radar_rf_kw > 0.0 or self.ais_active
        )

    @property
    def is_corridor_breach(self) -> bool:
        """True when the unit has left the tactical corridor into the hazard zone."""
        return self.corridor_deviation_m > CORRIDOR_HAZARD_M

    @property
    def is_speed_excursion(self) -> bool:
        """True when the unit is running above the nominal patrol envelope."""
        return self.speed_kts > PATROL_SPEED_CEILING_KTS


# --------------------------------------------------------------------------- #
# Simulation engine
# --------------------------------------------------------------------------- #


class MockTelemetryGenerator:
    """Deterministic telemetry source for a single unit under one scenario.

    Every scenario opens with nominal patrol frames and then, from
    ``onset_index`` onward, develops its characteristic signature. This keeps
    the state *transition* inside a short sliding window, which is what the
    semantic representation is built to describe.
    """

    #: Frames needed for a developing scenario to reach full amplitude.
    RAMP_FRAMES: int = 3

    def __init__(
        self,
        unit_id: str = "USV-GHOST-01",
        scenario: Scenario = Scenario.NOMINAL,
        *,
        seed: int = 1337,
        start_time: Optional[datetime] = None,
        origin: Tuple[float, float] = (36.6300, -121.9000),
        base_course_deg: float = 45.0,
        tick_seconds: float = 10.0,
        onset_index: int = 3,
    ) -> None:
        self.unit_id = unit_id
        self.scenario = Scenario(scenario)
        self.seed = seed
        self.start_time = start_time or datetime(2026, 9, 5, 14, 30, 0, tzinfo=timezone.utc)
        self.origin = origin
        self.base_course_deg = base_course_deg
        self.tick_seconds = tick_seconds
        self.onset_index = onset_index
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> None:
        """Rewind the generator to its initial position, time and RNG state."""
        self._rng = random.Random(self.seed)
        self._lat, self._lon = self.origin
        self._clock = self.start_time

    # -- public API --------------------------------------------------------- #

    def stream(self, count: int) -> Iterator[TacticalCommandEvent]:
        """Yield ``count`` consecutive telemetry frames."""
        for index in range(count):
            yield self._next_event(index)

    def generate(self, count: int) -> List[TacticalCommandEvent]:
        """Return ``count`` consecutive telemetry frames as a list."""
        return list(self.stream(count))

    # -- internals ---------------------------------------------------------- #

    def _next_event(self, index: int) -> TacticalCommandEvent:
        speed, course, emcon, radar_kw, ais, deviation = self._kinematics(index)
        event = TacticalCommandEvent(
            timestamp=self._clock.isoformat().replace("+00:00", "Z"),
            unit_id=self.unit_id,
            lat=round(self._lat, 6),
            lon=round(self._lon, 6),
            speed_kts=round(speed, 1),
            course_deg=round(course % 360.0, 1),
            emcon_state=emcon,
            radar_rf_kw=round(radar_kw, 1),
            ais_active=ais,
            corridor_deviation_m=round(deviation, 1),
        )
        self._advance(speed, course)
        self._clock += timedelta(seconds=self.tick_seconds)
        return event

    def _advance(self, speed_kts: float, course_deg: float) -> None:
        """Dead-reckon the unit forward one tick at the given speed and course."""
        distance_nm = speed_kts * (self.tick_seconds / 3600.0)
        heading = math.radians(course_deg)
        self._lat += (distance_nm / 60.0) * math.cos(heading)
        self._lon += (distance_nm / 60.0) * math.sin(heading) / max(
            math.cos(math.radians(self._lat)), 1e-6
        )

    def _progress(self, index: int) -> float:
        """Scenario development, 0.0 before onset through 1.0 at full amplitude."""
        if index < self.onset_index:
            return 0.0
        return min(1.0, (index - self.onset_index + 1) / float(self.RAMP_FRAMES))

    def _kinematics(
        self, index: int
    ) -> Tuple[float, float, EmconState, float, bool, float]:
        """Return ``(speed, course, emcon, radar_kw, ais, deviation)`` for a frame."""
        rng = self._rng
        # Baseline: silent patrol along the tactical fairway.
        speed = rng.uniform(12.0, 16.0)
        course = self.base_course_deg + rng.uniform(-2.0, 2.0)
        emcon = EmconState.ALPHA_SILENT
        radar_kw = 0.0
        ais = False
        deviation = rng.uniform(5.0, 45.0)

        progress = self._progress(index)
        if progress == 0.0 or self.scenario is Scenario.NOMINAL:
            return speed, course, emcon, radar_kw, ais, deviation

        if self.scenario is Scenario.EMCON_BREACH:
            # Surface-search radar and AIS come up while ALPHA remains declared.
            radar_kw = RADAR_BREACH_KW
            ais = True
            return speed, course, emcon, radar_kw, ais, deviation

        # NAV_DIVERGENCE: erratic 45-degree turn, sprint speed, corridor exit.
        course = self.base_course_deg + 45.0 * progress + rng.uniform(-6.0, 6.0)
        speed = 14.0 + (32.0 - 14.0) * progress + rng.uniform(-0.5, 0.5)
        deviation = 40.0 + (850.0 - 40.0) * progress + rng.uniform(-15.0, 15.0)
        return speed, course, emcon, radar_kw, ais, max(deviation, 0.0)


# --------------------------------------------------------------------------- #
# Text helpers for the semantic representation
# --------------------------------------------------------------------------- #

_EARTH_RADIUS_M = 6_371_000.0


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def _angular_delta(start_deg: float, end_deg: float) -> float:
    """Shortest signed rotation from ``start_deg`` to ``end_deg`` in (-180, 180]."""
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def _initial_bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def _numeric_transition(
    label: str,
    values: Sequence[float],
    unit: str,
    *,
    noise: float,
    spike: float,
    digits: int = 1,
) -> str:
    """Describe how a scalar channel moved across the window, in one sentence."""
    first, last = values[0], values[-1]
    delta = last - first
    peak, trough = max(values), min(values)
    mean = sum(values) / len(values)

    if abs(delta) < noise:
        if peak - trough >= spike:
            return (
                f"{label} unstable around {_fmt(mean, digits)}{unit} "
                f"(range {_fmt(trough, digits)}-{_fmt(peak, digits)}{unit})."
            )
        return f"{label} steady near {_fmt(mean, digits)}{unit}."

    if delta > 0:
        verb = "spiked" if delta >= spike else "increased"
    else:
        verb = "collapsed" if -delta >= spike else "decreased"

    sentence = f"{label} {verb} from {_fmt(first, digits)}{unit} to {_fmt(last, digits)}{unit}"
    if peak - max(first, last) >= spike:
        sentence += f", peaking at {_fmt(peak, digits)}{unit}"
    return sentence + "."


def _course_transition(values: Sequence[float]) -> str:
    """Describe course change, including whether the track was erratic."""
    first, last = values[0], values[-1]
    net = _angular_delta(first, last)
    swing = sum(abs(_angular_delta(a, b)) for a, b in zip(values, values[1:]))
    erratic = swing >= abs(net) * 1.5 + 15.0

    if abs(net) < 5.0:
        base = f"Course held near {_fmt(last, 0)}deg"
        return f"{base} with erratic yaw ({_fmt(swing, 0)}deg total swing)." if erratic else f"{base}."

    hand = "starboard" if net > 0 else "port"
    adjective = "erratic " if erratic else ""
    return (
        f"Course executed {adjective}{_fmt(abs(net), 0)}deg turn to {hand}, "
        f"from {_fmt(first, 0)}deg to {_fmt(last, 0)}deg."
    )


def _emcon_transition(states: Sequence[EmconState]) -> str:
    path: List[str] = []
    for state in states:
        if not path or path[-1] != state.value:
            path.append(state.value)
    if len(path) == 1:
        return f"EMCON state remained {path[0]}."
    return "EMCON state transitioned " + " -> ".join(path) + "."


def _ais_transition(flags: Sequence[bool]) -> str:
    first, last = flags[0], flags[-1]
    if first == last:
        state = "transmitting" if last else "silent"
        return f"AIS transponder remained {state}."
    return (
        "AIS transponder activated (silent -> transmitting)."
        if last
        else "AIS transponder secured (transmitting -> silent)."
    )


def _format_position(lat: float, lon: float) -> str:
    return f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'}/{abs(lon):.4f}{'E' if lon >= 0 else 'W'}"


# --------------------------------------------------------------------------- #
# Sliding window buffer
# --------------------------------------------------------------------------- #


class TacticalStateWindow:
    """Fixed-size sliding buffer of the most recent telemetry frames."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._frames: Deque[TacticalCommandEvent] = deque(maxlen=window_size)

    # -- container protocol ------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[TacticalCommandEvent]:
        return iter(self._frames)

    def __repr__(self) -> str:
        return f"TacticalStateWindow(window_size={self.window_size}, frames={len(self._frames)})"

    # -- mutation ----------------------------------------------------------- #

    def append(self, event: TacticalCommandEvent) -> None:
        """Push one frame, evicting the oldest once the window is full."""
        self._frames.append(event)

    def extend(self, events: Iterable[TacticalCommandEvent]) -> None:
        """Push many frames in order."""
        for event in events:
            self.append(event)

    def clear(self) -> None:
        self._frames.clear()

    # -- accessors ---------------------------------------------------------- #

    @property
    def frames(self) -> List[TacticalCommandEvent]:
        return list(self._frames)

    @property
    def oldest(self) -> Optional[TacticalCommandEvent]:
        return self._frames[0] if self._frames else None

    @property
    def latest(self) -> Optional[TacticalCommandEvent]:
        return self._frames[-1] if self._frames else None

    # -- semantic rendering ------------------------------------------------- #

    def to_semantic_representation(self) -> str:
        """Serialize the window into one embedding-friendly sentence block.

        The output is a single line of plain prose describing *transitions*
        rather than raw values, so that semantically similar situations (an
        EMCON breach today and one last week) land close together in vector
        space.
        """
        frames = self.frames
        if not frames:
            return "No telemetry frames buffered; unit state unknown."

        first, last = frames[0], frames[-1]
        span_s = (
            datetime.fromisoformat(last.timestamp.replace("Z", "+00:00"))
            - datetime.fromisoformat(first.timestamp.replace("Z", "+00:00"))
        ).total_seconds()

        parts: List[str] = [
            f"Unit {last.unit_id}: {len(frames)} telemetry frame{'' if len(frames) == 1 else 's'} spanning {span_s:.0f}s "
            f"from {first.timestamp} to {last.timestamp}.",
            _emcon_transition([f.emcon_state for f in frames]),
            _numeric_transition(
                "Radar power",
                [f.radar_rf_kw for f in frames],
                "kW",
                noise=0.05,
                spike=5.0,
            ),
            _ais_transition([f.ais_active for f in frames]),
            _numeric_transition(
                "Speed", [f.speed_kts for f in frames], "kts", noise=1.0, spike=8.0
            ),
            _course_transition([f.course_deg for f in frames]),
            _numeric_transition(
                "Route deviation",
                [f.corridor_deviation_m for f in frames],
                "m",
                noise=25.0,
                spike=200.0,
                digits=0,
            ),
            self._track_sentence(first, last),
            self._assessment_sentence(frames),
        ]
        return " ".join(parts)

    # -- rendering internals ------------------------------------------------ #

    @staticmethod
    def _track_sentence(
        first: TacticalCommandEvent, last: TacticalCommandEvent
    ) -> str:
        start, end = (first.lat, first.lon), (last.lat, last.lon)
        distance_m = _haversine_m(start, end)
        if distance_m < 1.0:
            return f"Track stationary at {_format_position(*end)}."
        return (
            f"Track advanced {distance_m:.0f}m on bearing "
            f"{_initial_bearing_deg(start, end):.0f}deg, "
            f"from {_format_position(*start)} to {_format_position(*end)}."
        )

    @staticmethod
    def _assessment_sentence(frames: Sequence[TacticalCommandEvent]) -> str:
        total = len(frames)
        findings: List[str] = []

        emitting = [f for f in frames if f.is_emcon_violation]
        if emitting:
            latest = emitting[-1]
            findings.append(
                f"EMCON ALPHA breach on {len(emitting)} of {total} frames -- "
                f"radar radiating {_fmt(latest.radar_rf_kw)}kW and AIS "
                f"{'transmitting' if latest.ais_active else 'secured'} under declared silence"
            )

        breaching = [f for f in frames if f.is_corridor_breach]
        if breaching:
            findings.append(
                f"corridor breach into hazard zone -- deviation "
                f"{_fmt(breaching[-1].corridor_deviation_m, 0)}m exceeds the "
                f"{_fmt(CORRIDOR_HAZARD_M, 0)}m limit"
            )
        elif any(f.corridor_deviation_m > CORRIDOR_WARNING_M for f in frames):
            findings.append(
                f"route drift beyond the {_fmt(CORRIDOR_WARNING_M, 0)}m advisory band"
            )

        sprinting = [f for f in frames if f.is_speed_excursion]
        if sprinting:
            findings.append(
                f"speed excursion -- {_fmt(sprinting[-1].speed_kts)}kts exceeds the "
                f"{_fmt(PATROL_SPEED_CEILING_KTS)}kts patrol ceiling"
            )

        if not findings:
            return "ASSESSMENT: nominal. Unit within patrol envelope, corridor and EMCON posture."
        return "ASSESSMENT: anomalous. Detected " + "; ".join(findings) + "."


# --------------------------------------------------------------------------- #
# Demonstration
# --------------------------------------------------------------------------- #


def _demo() -> None:
    window_size = 5
    frames_per_scenario = 7  # > window_size, so the buffer actually slides

    for offset, scenario in enumerate(Scenario):
        generator = MockTelemetryGenerator(
            unit_id="USV-GHOST-01",
            scenario=scenario,
            seed=1337 + offset,
        )
        window = TacticalStateWindow(window_size=window_size)
        window.extend(generator.stream(frames_per_scenario))

        print("=" * 78)
        print(f"SCENARIO: {scenario.value}   ({frames_per_scenario} frames emitted, {window})")
        print("=" * 78)
        print("-- buffered frames " + "-" * 59)
        for frame in window:
            print(
                f"  {frame.timestamp}  {frame.speed_kts:5.1f}kts  {frame.course_deg:5.1f}deg  "
                f"{frame.emcon_state.value:<16} radar={frame.radar_rf_kw:5.1f}kW  "
                f"ais={'ON ' if frame.ais_active else 'OFF'}  dev={frame.corridor_deviation_m:7.1f}m"
            )
        print("-- semantic representation (embedding input) " + "-" * 33)
        print(textwrap.fill(window.to_semantic_representation(), width=78))
        print()


if __name__ == "__main__":
    _demo()
