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


def _severity(p_closure: float, p_high: float, dur: float) -> tuple[str, float]:
    """Composite 0-1 severity score and tier label."""
    dur_norm = min((dur or 60) / 240.0, 1.0)  # cap influence at 4h
    score = 0.55 * p_closure + 0.30 * p_high + 0.15 * dur_norm
    tier = ("Critical" if score >= 0.6 else
            "High" if score >= 0.35 else
            "Moderate" if score >= 0.18 else "Low")
    return tier, score


def recommend_row(cause: str, p_closure: float, p_high: float, dur: float) -> dict:
    tier, score = _severity(p_closure, p_high, dur)

    base = {"Low": 1, "Moderate": 2, "High": 4, "Critical": 7}[tier]
    officers = base + (2 if p_high > 0.6 else 0) + (2 if (dur or 0) > 180 else 0)

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
