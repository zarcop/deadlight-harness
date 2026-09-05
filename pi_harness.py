"""Offline CPU-only Raspberry Pi harness. Run: python3 pi_harness.py"""
import hashlib
import json

EMCON_ALPHA = "ALPHA_SILENT"
CORRIDOR_HAZARD_M = 800.0
PATROL_SPEED_CEILING_KTS = 25.0

class RaspberryPiHarness:
    """Telemetry -> scripted proposals -> deterministic policy decisions."""
    def process(self, frames):
        if not frames:
            raise ValueError("At least one telemetry frame is required")
        latest = frames[-1]
        proposals = self.propose(latest)
        return {"unit_id": latest["unit_id"], "frame_count": len(frames),
                "proposals": proposals,
                "decisions": [self.evaluate(item, latest) for item in proposals]}

    def propose(self, frame):
        items = [self._proposal(frame, "HOLD_COURSE", {}, "Maintain patrol profile.")]
        alpha = frame["emcon_state"] == EMCON_ALPHA
        if alpha and (frame["radar_rf_kw"] > 0 or frame["ais_active"]):
            items += [
                self._proposal(frame, "CEASE_RADAR", {"radar_rf_kw": 0.0}, "Secure radar emission."),
                self._proposal(frame, "SECURE_AIS", {"ais_active": False}, "Secure AIS transmission."),
                self._proposal(frame, "ENABLE_RADAR", {"radar_rf_kw": 25.0}, "Denial-path test."),
            ]
        if frame["corridor_deviation_m"] > CORRIDOR_HAZARD_M:
            items.append(self._proposal(frame, "RETURN_TO_CORRIDOR", {"target_deviation_m": 0.0}, "Return to corridor."))
        if frame["speed_kts"] > PATROL_SPEED_CEILING_KTS:
            items += [
                self._proposal(frame, "REDUCE_SPEED", {"target_speed_kts": PATROL_SPEED_CEILING_KTS}, "Reduce speed."),
                self._proposal(frame, "INCREASE_SPEED", {"target_speed_kts": frame["speed_kts"] + 5.0}, "Denial-path test."),
            ]
        return items

    def evaluate(self, proposal, frame):
        safe = {"HOLD_COURSE", "CEASE_RADAR", "SECURE_AIS", "RETURN_TO_CORRIDOR", "REDUCE_SPEED"}
        if proposal["action"] in safe:
            return self._decision(proposal, "ALLOW", "SAFE_RECOVERY", "Action maintains or restores the approved envelope.")
        if proposal["action"] == "ENABLE_RADAR" and frame["emcon_state"] == EMCON_ALPHA:
            return self._decision(proposal, "DENY", "EMCON_ALPHA", "Emission is forbidden under EMCON ALPHA.")
        if proposal["action"] == "INCREASE_SPEED" and frame["speed_kts"] >= PATROL_SPEED_CEILING_KTS:
            return self._decision(proposal, "DENY", "PATROL_SPEED_CEILING", "Proposal exceeds the patrol speed ceiling.")
        return self._decision(proposal, "DENY", "UNAUTHORIZED_ACTION", "Action is not authorized locally.")

    @staticmethod
    def _proposal(frame, action, arguments, rationale):
        data = json.dumps({"timestamp": frame["timestamp"], "unit_id": frame["unit_id"], "action": action, "arguments": arguments}, sort_keys=True)
        return {"proposal_id": hashlib.sha256(data.encode()).hexdigest()[:16], "action": action, "arguments": arguments, "rationale": rationale}

    @staticmethod
    def _decision(proposal, decision, rule, reason):
        return {"proposal_id": proposal["proposal_id"], "decision": decision, "rule": rule, "reason": reason}

if __name__ == "__main__":
    sample = {"timestamp": "2026-09-05T14:30:00Z", "unit_id": "USV-GHOST-01", "emcon_state": "ALPHA_SILENT", "radar_rf_kw": 25.0, "ais_active": True, "speed_kts": 14.0, "corridor_deviation_m": 20.0}
    print(json.dumps(RaspberryPiHarness().process([sample]), indent=2))
