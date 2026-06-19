import pandas as pd, numpy as np
f = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
df = pd.read_csv(f, low_memory=False).replace('NULL', np.nan)
df['start'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True).dt.tz_convert('Asia/Kolkata')
df['rc'] = df['requires_road_closure'].astype(str).str.upper().eq('TRUE')
df['high'] = df['priority'].eq('High')

print("hour-of-day event counts:")
print(df['start'].dt.hour.value_counts().sort_index().to_string())
print("\nday-of-week (0=Mon):")
print(df['start'].dt.dayofweek.value_counts().sort_index().to_string())

print("\nroad_closure rate by event_cause:")
print((df.groupby('event_cause')['rc'].agg(['mean','count']).sort_values('mean',ascending=False)).round(3).to_string())

print("\nroad_closure rate by event_type:")
print(df.groupby('event_type')['rc'].agg(['mean','count']).round(3).to_string())

print("\nhigh-priority rate by event_cause:")
print((df.groupby('event_cause')['high'].agg(['mean','count']).sort_values('mean',ascending=False)).round(3).to_string())

# planned event lead time (created vs start)
df['created'] = pd.to_datetime(df['created_date'], errors='coerce', utc=True)
df['startu'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
lead = (df['startu'] - df['created']).dt.total_seconds()/3600
print("\nlead time hours (start - created) by event_type:")
print(df.assign(lead=lead).groupby('event_type')['lead'].describe(percentiles=[.5,.9]).round(2).to_string())

print("\nevents per day overall:", round(len(df)/ ((df['start'].max()-df['start'].min()).days),1))
print("corridor x rc top:")
print(df.groupby('corridor')['rc'].agg(['mean','count']).sort_values('count',ascending=False).head(8).round(3).to_string())
