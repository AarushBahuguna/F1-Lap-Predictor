import fastf1
import os

print("FastF1 Version:", fastf1.__version__)
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

session = fastf1.get_session(2023, 1, 'R')
session.load(weather=True)
laps = session.laps.pick_quicklaps()
weather = laps.get_weather_data()

print(f"Laps: {len(laps)}, Weather: {len(weather)}")
print("Sample Lap:", laps.iloc[0][['LapTime', 'Compound', 'TyreLife', 'LapNumber']])
print("Sample Weather:", weather.iloc[0][['AirTemp', 'TrackTemp', 'Humidity']])
