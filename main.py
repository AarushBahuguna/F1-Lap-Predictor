import os
import argparse
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Set up FastF1 cache
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

class LapTimePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.encoders = {
            'Compound': LabelEncoder(),
            'Track': LabelEncoder()
        }
        self.features = ['Track', 'Compound', 'TyreLife', 'AirTemp', 'TrackTemp', 'Humidity', 'LapNumber']
    
    def fetch_training_data(self, year=2024):
        # Fetch data for a few races to build a dataset
        races = [1, 2, 3, 4, 5]
        all_laps = []
        
        for round_num in races:
            try:
                print(f"Fetching {year} Round {round_num} data...")
                session = fastf1.get_session(year, round_num, 'R')
                session.load(weather=True)
                
                track_name = session.event['EventName']
                
                laps = session.laps
                
                # Filter for useful laps (ignoring in/out laps, and anomalous laps)
                # Ensure the lap was timed and IsAccurate is True, which usually excludes Safety Car laps
                laps = laps.pick_quicklaps()
                
                # Get weather data
                weather_data = laps.get_weather_data()
                
                # Reset indices for proper alignment in DataFrame
                laps = laps.reset_index(drop=True)
                weather_data = weather_data.reset_index(drop=True)
                
                # Drop rows where variables might be missing
                temp_df = pd.DataFrame({
                    'Track': track_name,
                    'Compound': laps['Compound'],
                    'TyreLife': laps['TyreLife'],
                    'LapNumber': laps['LapNumber'],
                    'AirTemp': weather_data['AirTemp'],
                    'TrackTemp': weather_data['TrackTemp'],
                    'Humidity': weather_data['Humidity'],
                    'LapTime': laps['LapTime'].dt.total_seconds()
                })
                
                all_laps.append(temp_df)
                print(f"Added {len(temp_df)} laps from {track_name}")
            except Exception as e:
                print(f"Error loading round {round_num}: {e}")
                
        df = pd.concat(all_laps, ignore_index=True)
        return df.dropna()
        
    def prepare_data(self, df):
        df_encoded = df.copy()
        
        # Fit and transform categorical variables
        df_encoded['Track'] = self.encoders['Track'].fit_transform(df_encoded['Track'])
        df_encoded['Compound'] = self.encoders['Compound'].fit_transform(df_encoded['Compound'])
        
        X = df_encoded[self.features]
        y = df_encoded['LapTime']
        
        return X, y
        
    def train(self, year=2024):
        print("Fetching data...")
        df = self.fetch_training_data(year)
        
        print(f"Collected {len(df)} laps for training.")
        if len(df) == 0:
            print("No data collected!")
            return
            
        X, y = self.prepare_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model Mean Absolute Error: {mae:.3f} seconds")
        
        self.save_model()
        
    def save_model(self):
        joblib.dump({'model': self.model, 'encoders': self.encoders}, 'predictor_model.joblib')
        print("Model saved to predictor_model.joblib")
        
    def load_model(self):
        data = joblib.load('predictor_model.joblib')
        self.model = data['model']
        self.encoders = data['encoders']
        
    def predict(self, track, compound, tyre_life, air_temp, track_temp, humidity, lap_number):
        try:
            track_enc = self.encoders['Track'].transform([track])[0]
        except ValueError:
            print(f"Warning: Track '{track}' not seen in training. Using default (0).")
            track_enc = 0
            
        try:
            compound_enc = self.encoders['Compound'].transform([compound])[0]
        except ValueError:
            print(f"Warning: Compound '{compound}' not seen in training. Using default (0).")
            compound_enc = 0
            
        data = pd.DataFrame({
            'Track': [track_enc],
            'Compound': [compound_enc],
            'TyreLife': [tyre_life],
            'AirTemp': [air_temp],
            'TrackTemp': [track_temp],
            'Humidity': [humidity],
            'LapNumber': [lap_number]
        })
        
        pred = self.model.predict(data)[0]
        return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Lap Time Predictor")
    parser.add_argument("--train", action="store_true", help="Force retrain the model")
    args = parser.parse_args()
    
    predictor = LapTimePredictor()
    
    model_exists = os.path.exists('predictor_model.joblib')
    
    if args.train or not model_exists:
        print("Training new model. This may take a few minutes as data is downloaded...")
        predictor.train(year=2024)
    else:
        predictor.load_model()
        print("Loaded existing model.")
    
    if not args.train:
        print("\n=== F1 Lap Time Predictor ===")
        print("Enter parameters to predict lap time (press Enter to use defaults):")
        
        track = input("Track (default: Bahrain Grand Prix): ") or "Bahrain Grand Prix"
        compound = input("Tire Compound [SOFT/MEDIUM/HARD] (default: SOFT): ").upper() or "SOFT"
        
        try:
            tyre_life = float(input("Tyre Life in laps (default: 5.0): ") or 5.0)
            air_temp = float(input("Air Temperature °C (default: 25.0): ") or 25.0)
            track_temp = float(input("Track Temperature °C (default: 32.0): ") or 32.0)
            humidity = float(input("Humidity % (default: 45.0): ") or 45.0)
            lap_number = int(input("Lap Number / Fuel Load proxy (default: 15): ") or 15)
        except ValueError:
            print("Invalid numeric input. Using defaults.")
            tyre_life, air_temp, track_temp, humidity, lap_number = 5.0, 25.0, 32.0, 45.0, 15
            
        print("\n--- Model Prediction Input ---")
        print(f"  Track      : {track}")
        print(f"  Compound   : {compound}")
        print(f"  Tyre Life  : {tyre_life:.1f} laps")
        print(f"  Air Temp   : {air_temp}°C")
        print(f"  Track Temp : {track_temp}°C")
        print(f"  Humidity   : {humidity}%")
        print(f"  Lap Number : {lap_number} (Fuel impact)")
        
        try:
            prediction = predictor.predict(track, compound, tyre_life, air_temp, track_temp, humidity, lap_number)
            
            mins = int(prediction // 60)
            secs = prediction % 60
            print(f"\n🏎️  => Predicted Lap Time: {mins}:{secs:06.3f} ({prediction:.3f} seconds)\n")
        except Exception as e:
            print(f"Prediction failed: {e}")
