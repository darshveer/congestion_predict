"""Load and clean the raw traffic-event log into a tidy, analysis-ready frame."""
import numpy as np
import pandas as pd

from . import DATA_FILE

# Free-text breakdown reasons are noisy; we only normalise the columns we model on.
_CAUSE_CANON = {
    "debris": "debris",
    "fog / low visibility": "fog",
    "test_demo": "others",  # internal test rows -> fold into 'others'
}

# Causes that are genuinely "event-driven" gatherings (the problem statement's focus).
GATHERING_CAUSES = {"public_event", "procession", "vip_movement", "protest"}


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_raw(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # The export uses the literal string 'NULL' for missing values.
    return df.replace("NULL", np.nan)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- timestamps (stored UTC; we reason in IST for human/temporal features) ---
    for c in ["start_datetime", "end_datetime", "created_date",
              "resolved_datetime", "closed_datetime", "modified_datetime"]:
        df[c] = _to_dt(df[c])
    df["start"] = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    df = df[df["start"].notna()].copy()

    # --- categorical normalisation ---
    df["event_cause"] = (df["event_cause"].astype(str).str.strip().str.lower()
                         .replace(_CAUSE_CANON))
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df["corridor"] = df["corridor"].fillna("Non-corridor").astype(str).str.strip()
    df["veh_type"] = df["veh_type"].fillna("unknown").astype(str).str.strip().str.lower()
    df["zone"] = df["zone"].fillna("unknown").astype(str).str.strip()
    df["priority"] = df["priority"].fillna("Low").astype(str).str.strip()

    # --- targets ---
    # Road closure -> barricading / diversion need.
    df["road_closure"] = df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int)
    # High priority -> manpower urgency.
    df["high_priority"] = df["priority"].eq("High").astype(int)
    df["is_gathering"] = df["event_cause"].isin(GATHERING_CAUSES).astype(int)

    # --- clearance duration (resource deployment time), heavily cleaned ---
    end = (df["end_datetime"].fillna(df["resolved_datetime"]).fillna(df["closed_datetime"]))
    dur = (end - df["start_datetime"]).dt.total_seconds() / 60.0
    # Keep only physically plausible clearances (0 min .. 24 h); rest is censored/garbage.
    df["duration_min"] = dur.where((dur > 0) & (dur <= 24 * 60))

    df = df.sort_values("start").reset_index(drop=True)
    return df


def load_clean(path: str = DATA_FILE) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    d = load_clean()
    print("clean shape:", d.shape)
    print("road_closure rate:", d["road_closure"].mean().round(3))
    print("high_priority rate:", d["high_priority"].mean().round(3))
    print("duration coverage:", d["duration_min"].notna().mean().round(3))
    print("date span:", d["start"].min(), "->", d["start"].max())
