import pandas as pd, numpy as np
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60)

f = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
df = pd.read_csv(f, low_memory=False)
print("SHAPE:", df.shape)
print("\nCOLUMNS:", list(df.columns))

# Replace string NULLs
df = df.replace('NULL', np.nan)

print("\n--- NULL fraction per column ---")
print((df.isna().mean().sort_values()).round(3).to_string())

for c in ['event_type','event_cause','status','requires_road_closure','priority','corridor',
          'veh_type','zone','direction','authenticated','reason_breakdown']:
    print(f"\n--- {c} ({df[c].nunique()} unique) ---")
    print(df[c].value_counts(dropna=False).head(12).to_string())

# Datetime parse
for c in ['start_datetime','end_datetime','created_date','resolved_datetime','closed_datetime','modified_datetime']:
    df[c] = pd.to_datetime(df[c], errors='coerce', utc=True)

print("\n--- start_datetime range ---")
print(df['start_datetime'].min(), "->", df['start_datetime'].max())

# Duration: how long events last (target signal for congestion impact)
end = df['end_datetime'].fillna(df['resolved_datetime']).fillna(df['closed_datetime'])
dur = (end - df['start_datetime']).dt.total_seconds()/60.0
print("\n--- event duration minutes (start->resolved/closed) ---")
print(dur.describe(percentiles=[.25,.5,.75,.9,.95]).round(1).to_string())
print("valid durations:", dur.notna().sum(), "/", len(df))

# spatial
print("\n--- lat/lon ---")
print(df[['latitude','longitude']].describe().round(4).to_string())
print("zones:", df['zone'].nunique(), "junctions:", df['junction'].nunique())
