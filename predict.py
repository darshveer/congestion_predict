"""Score new/incoming events and emit an operational deployment plan.

Trains the ImpactModel on the full history, then scores example incoming events.
Replace `incoming` with a live feed (same schema as the raw export) in production.
"""
import warnings
import pandas as pd

from congestion import data as D
from congestion.models import ImpactModel
from congestion import recommend as R

warnings.filterwarnings("ignore", category=UserWarning)

# Example incoming events (minimal schema the model needs to score an event).
INCOMING = pd.DataFrame([
    dict(id="NEW1", event_type="planned", event_cause="public_event", corridor="Hosur Road",
         veh_type=None, latitude=12.915, longitude=77.622, requires_road_closure=None,
         priority=None, authenticated="yes", start_datetime="2024-04-10 18:30:00+00",
         zone="unknown"),
    dict(id="NEW2", event_type="unplanned", event_cause="accident", corridor="ORR East 1",
         veh_type="heavy_vehicle", latitude=12.935, longitude=77.690, requires_road_closure=None,
         priority=None, authenticated="yes", start_datetime="2024-04-10 09:15:00+00",
         zone="unknown"),
    dict(id="NEW3", event_type="unplanned", event_cause="vehicle_breakdown", corridor="Non-corridor",
         veh_type="lcv", latitude=12.972, longitude=77.594, requires_road_closure=None,
         priority=None, authenticated="yes", start_datetime="2024-04-10 14:00:00+00",
         zone="unknown"),
])


def _prep(raw: pd.DataFrame) -> pd.DataFrame:
    """Run incoming rows through the same cleaning the model was trained on."""
    for col in D.load_raw().columns:
        if col not in raw.columns:
            raw[col] = None
    return D.clean(raw)


def main():
    hist = D.load_clean()
    model = ImpactModel().fit(hist)

    events = _prep(INCOMING.copy())
    preds = model.predict(events)
    recs = R.recommend(events, preds)
    out = pd.concat([events[["id", "event_cause", "corridor"]], preds, recs], axis=1)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    for _, r in out.iterrows():
        print(f"\n=== {r['id']}: {r['event_cause']} on {r['corridor']} ===")
        print(f"  P(road closure)={r['p_road_closure']:.2f}  "
              f"P(high/corridor)={r['p_high_priority']:.2f}  "
              f"expected clearance ~{r['exp_duration_min']:.0f} min")
        print(f"  SEVERITY: {r['severity_tier']} ({r['severity_score']})")
        print(f"  DEPLOY  : {r['rec_officers']} officers, {r['rec_barricades']} barricades, "
              f"tow-crane={r['rec_tow_crane']}")
        print(f"  TRAFFIC : {r['rec_diversion']}")


if __name__ == "__main__":
    main()
