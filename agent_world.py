"""The mission world an agent operates in.

Telemetry says what the vessel is doing. This module says what is around it:
the route it has been tasked to follow, the surface contacts sharing the water,
its remaining endurance, and the quality of its links.

The world is persistent and causal. Contacts keep their identity and move on
their own courses; waypoints are fixed positions the unit must actually reach;
battery drains with the speed the agent commands and the emitters it switches
on. An agent that steers toward a waypoint sees the distance close. That is the
minimum an agent needs to reason about consequences -- a world that reshuffles
itself every step can only be reacted to, never planned against.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

MISSION_BRIEF = (
    "Patrol the approved route corridor under EMCON ALPHA (emissions silent), "
    "visiting each waypoint in order, keeping clear of surface contacts, and "
    "escalating unsafe or ambiguous states to the operator."
)

NM_PER_DEG_LAT = 60.0
CONTACT_CAUTION_NM = 1.2
WAYPOINT_ARRIVAL_NM = 0.3


def _nm_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Flat-earth distance in nautical miles; accurate at these ranges."""
    dlat = (b[0] - a[0]) * NM_PER_DEG_LAT
    dlon = (b[1] - a[1]) * NM_PER_DEG_LAT * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot(dlat, dlon)


def _bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dlat = (b[0] - a[0]) * NM_PER_DEG_LAT
    dlon = (b[1] - a[1]) * NM_PER_DEG_LAT * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.degrees(math.atan2(dlon, dlat)) % 360.0


def _offset(origin: Tuple[float, float], bearing_deg: float, nm: float) -> Tuple[float, float]:
    rad = math.radians(bearing_deg)
    lat = origin[0] + (nm * math.cos(rad)) / NM_PER_DEG_LAT
    lon = origin[1] + (nm * math.sin(rad)) / (NM_PER_DEG_LAT * math.cos(math.radians(origin[0])))
    return lat, lon


@dataclass
class Contact:
    """A surface vessel sharing the water with the unit."""

    contact_id: str
    lat: float
    lon: float
    course_deg: float
    speed_kts: float
    classification: str

    def advance(self, seconds: float) -> None:
        self.lat, self.lon = _offset(
            (self.lat, self.lon), self.course_deg, self.speed_kts * seconds / 3600.0
        )


@dataclass
class MissionWorld:
    """Persistent surroundings for one patrol, advanced once per committed step."""

    route: List[Tuple[float, float]]
    contacts: List[Contact]
    seed: int
    tick_seconds: float = 10.0
    waypoint_index: int = 0
    battery_pct: float = 92.0
    completed_waypoints: int = 0
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    # -- construction --------------------------------------------------------- #

    @classmethod
    def create(
        cls,
        origin: Tuple[float, float],
        *,
        base_course_deg: float = 45.0,
        seed: int = 2026,
        tick_seconds: float = 10.0,
    ) -> "MissionWorld":
        """Lay out a route along the base course and seed a few contacts."""
        rng = random.Random(seed)
        route: List[Tuple[float, float]] = []
        point = origin
        for leg in range(5):
            bearing = base_course_deg + rng.uniform(-12.0, 12.0)
            point = _offset(point, bearing, 0.7 + leg * 0.05)
            route.append(point)

        classes = ("fishing vessel", "merchant", "unknown small craft")
        contacts = []
        for index in range(3):
            along = rng.uniform(0.6, 3.0)
            beam = rng.choice((-1, 1)) * rng.uniform(0.4, 1.6)
            anchor = _offset(origin, base_course_deg, along)
            lat, lon = _offset(anchor, base_course_deg + 90.0, beam)
            contacts.append(Contact(
                contact_id=f"SURF-{index + 1:02d}",
                lat=lat, lon=lon,
                course_deg=rng.uniform(0.0, 360.0),
                speed_kts=rng.uniform(2.0, 9.0),
                classification=classes[index % len(classes)],
            ))
        return cls(route=route, contacts=contacts, seed=seed,
                   tick_seconds=tick_seconds, _rng=rng)

    # -- dynamics ------------------------------------------------------------- #

    def advance(self, position: Tuple[float, float], speed_kts: float, radar_kw: float) -> None:
        """Move the world one tick forward after the unit's committed step."""
        for contact in self.contacts:
            contact.advance(self.tick_seconds)

        # Endurance: hotel load, plus propulsion that grows with the square of
        # speed, plus the radar if it is radiating. A sprint costs visibly more.
        drain = 0.08 + 0.0012 * speed_kts ** 2 + (0.15 if radar_kw > 0 else 0.0)
        self.battery_pct = max(0.0, self.battery_pct - drain)

        if self.waypoint_index < len(self.route) and (
            _nm_between(position, self.route[self.waypoint_index]) <= WAYPOINT_ARRIVAL_NM
        ):
            self.waypoint_index += 1
            self.completed_waypoints += 1

    def link_quality(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """(comms, gps) quality. A jamming area sits past the second waypoint."""
        if len(self.route) < 3:
            return 0.92, 0.94
        centre = self.route[2]
        distance = _nm_between(position, centre)
        degradation = max(0.0, 1.0 - distance / 0.8)
        return round(0.92 - 0.55 * degradation, 2), round(0.94 - 0.6 * degradation, 2)

    # -- observation ---------------------------------------------------------- #

    def contacts_view(self, position: Tuple[float, float], course_deg: float) -> List[Dict[str, object]]:
        """Contacts sorted by range, with a closest-point-of-approach estimate."""
        view = []
        for c in self.contacts:
            rng_nm = _nm_between(position, (c.lat, c.lon))
            view.append({
                "contact_id": c.contact_id,
                "classification": c.classification,
                "range_nm": round(rng_nm, 2),
                "bearing_deg": round(_bearing_deg(position, (c.lat, c.lon)), 1),
                "relative_bearing_deg": round(
                    (_bearing_deg(position, (c.lat, c.lon)) - course_deg) % 360.0, 1
                ),
                "course_deg": round(c.course_deg, 1),
                "speed_kts": round(c.speed_kts, 1),
                "inside_caution_range": rng_nm < CONTACT_CAUTION_NM,
            })
        return sorted(view, key=lambda item: item["range_nm"])

    def context(
        self,
        position: Tuple[float, float],
        course_deg: float,
        speed_kts: float,
    ) -> Dict[str, object]:
        """The mission facts a command agent observes this step.

        Keeps the flat keys the deterministic brain reads, and adds the richer
        structure the Claude agent reasons over.
        """
        contacts = self.contacts_view(position, course_deg)
        closest = contacts[0] if contacts else None
        comms, gps = self.link_quality(position)

        done = self.waypoint_index >= len(self.route)
        target = self.route[-1] if done else self.route[self.waypoint_index]
        distance = _nm_between(position, target)
        eta_min = (distance / speed_kts * 60.0) if speed_kts > 0.5 else None

        if done:
            phase = "return_to_base"
        elif self.battery_pct < 25.0:
            phase = "endurance_critical"
        elif gps < 0.6:
            phase = "degraded_navigation"
        else:
            phase = "patrol"

        return {
            "mission_phase": phase,
            "route_progress": f"{self.completed_waypoints}/{len(self.route)} waypoints",
            "active_waypoint_id": f"WP-{min(self.waypoint_index, len(self.route) - 1) + 1}",
            "active_waypoint_index": min(self.waypoint_index, len(self.route) - 1) + 1,
            "distance_to_waypoint_nm": round(distance, 2),
            "bearing_to_waypoint_deg": round(_bearing_deg(position, target), 1),
            "eta_minutes": None if eta_min is None else round(eta_min, 1),
            "next_waypoint_lat": round(target[0], 6),
            "next_waypoint_lon": round(target[1], 6),
            "closest_contact_id": closest["contact_id"] if closest else None,
            "closest_contact_range_nm": closest["range_nm"] if closest else 99.0,
            "closest_contact_bearing_deg": closest["bearing_deg"] if closest else None,
            "contacts": contacts,
            "comms_quality": comms,
            "gps_quality": gps,
            "battery_pct": round(self.battery_pct, 1),
            "sensor_mode": "PASSIVE",
        }

    def snapshot(self) -> Dict[str, object]:
        """Positions for a display: route, contacts, progress."""
        return {
            "route": [{"lat": round(p[0], 6), "lon": round(p[1], 6)} for p in self.route],
            "waypoint_index": self.waypoint_index,
            "contacts": [
                {"id": c.contact_id, "lat": round(c.lat, 6), "lon": round(c.lon, 6)}
                for c in self.contacts
            ],
            "battery_pct": round(self.battery_pct, 1),
        }


def distance_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return _nm_between(a, b)


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return _bearing_deg(a, b)

