"""Turn model predictions into concrete operational recommendations.

The problem statement asks for *manpower, barricading, and diversion* plans. We map the
predicted impact (closure probability, priority, expected clearance time, cause) to a
deployment package using transparent, auditable rules anchored on the model outputs.
This layer is intentionally interpretable so field officers can trust/override it.
"""
from __future__ import annotations
import pandas as pd

# Causes that typically need physical lane management when impact is high.
_BARRICADE_CAUSES = {"public_event", "procession", "vip_movement", "protest",
                     "construction", "tree_fall", "accident", "water_logging", "debris"}


# Severity weights & thresholds tuned in tune_severity.py: chosen so the score *ranks*
# events by their RECORDED impact (real road closures + clearance times) on held-out data.
# This replaced an earlier formula that gave 30% weight to the administrative "is-on-a-
# corridor" flag, which made almost every event look "High". With these weights the actual
# closure rate rises cleanly across tiers: Low ~1% -> Moderate ~7% -> High ~18% -> Critical ~55%.
_W_CLOSURE, _W_DURATION, _W_CAUSE = 0.55, 0.35, 0.10
_T_MODERATE, _T_HIGH, _T_CRITICAL = 0.10, 0.20, 0.46


def _severity(p_closure: float, dur: float, cause: str) -> tuple[str, float]:
    """Composite 0-1 severity score and tier label, anchored on genuine impact signals."""
    dur_norm = min((dur or 60) / 240.0, 1.0)               # expected clearance, capped at 4h
    cause_flag = 1.0 if cause in _BARRICADE_CAUSES else 0.0  # disruptive cause
    score = _W_CLOSURE * p_closure + _W_DURATION * dur_norm + _W_CAUSE * cause_flag

    # Operational guard: a likely road closure is severe on its own, whatever the blend.
    if p_closure >= 0.80 or score >= _T_CRITICAL:
        tier = "Critical"
    elif p_closure >= 0.60 or score >= _T_HIGH:
        tier = "High"
    elif score >= _T_MODERATE:
        tier = "Moderate"
    else:
        tier = "Low"
    return tier, score


def recommend_row(cause: str, p_closure: float, p_high: float, dur: float) -> dict:
    tier, score = _severity(p_closure, dur, cause)

    base = {"Low": 1, "Moderate": 2, "High": 4, "Critical": 7}[tier]
    # extra officers for genuine closure risk and long expected clearance (not corridor flag)
    officers = base + (2 if p_closure > 0.4 else 0) + (2 if (dur or 0) > 180 else 0)

    needs_barricade = (p_closure > 0.35) or (tier in ("High", "Critical") and cause in _BARRICADE_CAUSES)
    barricades = 0
    if needs_barricade:
        barricades = {"Moderate": 2, "High": 4, "Critical": 8}.get(tier, 2)

    if p_closure > 0.6:
        diversion = "Full diversion: close affected stretch, signed detour + upstream signal hold"
    elif p_closure > 0.35:
        diversion = "Partial diversion: lane drop, merge control at nearest junction"
    elif tier in ("High", "Critical"):
        diversion = "Advisory diversion: VMS warning, encourage alternate route"
    else:
        diversion = "No diversion: monitor and keep lane open"

    return {
        "severity_tier": tier,
        "severity_score": round(float(score), 3),
        "rec_officers": int(officers),
        "rec_barricades": int(barricades),
        "rec_diversion": diversion,
        "rec_tow_crane": bool(p_closure > 0.4 or (dur or 0) > 180),
        "rec_eta_minutes": int(dur) if dur == dur else None,  # NaN-safe
    }


def recommend(events: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """events: cleaned rows (needs 'event_cause'); preds: ImpactModel.predict output."""
    rows = []
    for idx in events.index:
        rows.append(recommend_row(
            events.at[idx, "event_cause"],
            preds.at[idx, "p_road_closure"],
            preds.at[idx, "p_high_priority"],
            preds.at[idx, "exp_duration_min"]))
    return pd.DataFrame(rows, index=events.index)
