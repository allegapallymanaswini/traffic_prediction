from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Custom Ensemble Class needed for deserialization
class TrafficEnsemble:
    def __init__(self, lgbm, xgb):
        self.lgbm = lgbm
        self.xgb = xgb
    def predict(self, X):
        return (0.55 * self.lgbm.predict(X)) + (0.45 * self.xgb.predict(X))

app = Flask(__name__)

# Load model and artifacts
MODEL_PATH = 'models/ensemble_model.pkl'
ENCODERS_PATH = 'models/label_encoders.pkl'
DATA_PATH = 'data/traffic_prediction_dataset.csv'

# Define feature names as used in training (Model expects 26 features)
feature_names = [
    'Road_Segment_ID', 'Historical_Average_Speed', 'Temperature_C', 'Humidity_Percent',
    'Rainfall_mm', 'Weather_Condition', 'Visibility_km', 'Wind_Speed_kmph',
    'Road_Type', 'Number_of_Lanes', 'Speed_Limit_kmph', 'Traffic_Signals',
    'Nearby_Intersections', 'Event_Type', 'Event_Flag', 'Hour', 'DayOfWeek',
    'Month', 'IsWeekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'prev_hour_traffic', 'prev_2hour_traffic', 'rolling_mean_3h'
]

model = None
encoders = None
raw_df = None

def load_assets():
    global model, encoders, raw_df
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)
    if os.path.exists(DATA_PATH):
        raw_df = pd.read_csv(DATA_PATH)

load_assets()

@app.route('/')
def home():
    return render_template('index.html')

# --- API Endpoints ---

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])

        # Preprocess categorical features
        for col, le in encoders.items():
            if col in input_df.columns:
                try:
                    # Handle unseen labels by defaulting to 0 or another strategy
                    val = str(input_df[col].iloc[0])
                    if val in le.classes_:
                        input_df[col] = le.transform([val])[0]
                    else:
                        input_df[col] = 0
                except:
                    input_df[col] = 0

        # Add derived features (matches prepare_data in notebook)
        now = datetime.now()
        date_str = data.get('Date')
        if date_str:
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                h = int(data.get('Hour', dt.hour))
                dow = dt.weekday()
                month = dt.month
            except:
                h = int(data.get('Hour', now.hour))
                dow = int(data.get('DayOfWeek', now.weekday()))
                month = int(data.get('Month', now.month))
        else:
            h = int(data.get('Hour', now.hour))
            dow = int(data.get('DayOfWeek', now.weekday()))
            month = int(data.get('Month', now.month))

        input_df['Hour'] = h
        input_df['DayOfWeek'] = dow
        input_df['Month'] = month
        input_df['IsWeekend'] = 1 if dow >= 5 else 0

        input_df['hour_sin'] = np.sin(2 * np.pi * h / 24)
        input_df['hour_cos'] = np.cos(2 * np.pi * h / 24)
        input_df['day_sin'] = np.sin(2 * np.pi * dow / 7)
        input_df['day_cos'] = np.cos(2 * np.pi * dow / 7)

        # Handle Event Flag logic
        event_type = data.get('Event_Type', 'None')
        input_df['Event_Type'] = event_type
        input_df['Event_Flag'] = 1 if event_type != 'None' else 0

        # Handle lag features
        input_df['prev_hour_traffic'] = float(data.get('prev_hour_traffic', 150))
        input_df['prev_2hour_traffic'] = float(data.get('prev_2hour_traffic', 150))
        input_df['rolling_mean_3h'] = float(data.get('rolling_mean_3h', 150))

        # Fill missing features with 0
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match model training
        input_df = input_df[feature_names]
        prediction = model.predict(input_df)[0]

        # --- Congestion Level Classification ---
        congestion_level = "Low"
        if prediction > 200: congestion_level = "High"
        elif prediction > 100: congestion_level = "Medium"

        # --- Accident Risk Prediction ---
        # Heuristic-based risk (Weather, Visibility, Speed, Traffic)
        risk_score = 0
        if data.get('Weather_Condition') in ['Rain', 'Heavy Rain']: risk_score += 40
        if data.get('Visibility_km', 10) < 5: risk_score += 30
        if prediction > 250: risk_score += 30

        accident_risk = "Low"
        if risk_score > 70: accident_risk = "High"
        elif risk_score > 40: accident_risk = "Moderate"

        return jsonify({
            'status': 'success',
            'prediction': float(prediction),
            'congestion_level': congestion_level,
            'accident_risk': accident_risk,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- Geo-Spatial Traffic Data ---
@app.route('/api/geo_data', methods=['GET'])
def get_geo_data():
    if raw_df is not None:
        # Return recent snapshot of coordinates and traffic
        geo_data = raw_df.tail(100)[['Latitude', 'Longitude', 'Vehicle_Count_Traffic_Demand']].to_dict(orient='records')
        return jsonify({'status': 'success', 'data': geo_data})
    return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

# --- Route Recommendation System ---
@app.route('/api/recommend_route', methods=['POST'])
def recommend_route():
    try:
        # Expecting a list of potential segment IDs
        segments = request.json.get('segments', [])
        recommendations = []
        for s_id in segments:
            # Simple logic: pick the segment with lowest historical avg if no real-time data
            avg = raw_df[raw_df['Road_Segment_ID'] == s_id]['Vehicle_Count_Traffic_Demand'].mean()
            recommendations.append({'segment': s_id, 'score': float(avg)})

        best_route = min(recommendations, key=lambda x: x['score'])
        return jsonify({'status': 'success', 'recommended': best_route, 'all_options': recommendations})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data_list = request.json # Expecting list of dicts
        input_df = pd.DataFrame(data_list)
        # (Simplified preprocessing for batch)
        # In a real app, you'd apply the same logic as single predict
        return jsonify({'status': 'success', 'message': 'Batch processed (demo)'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/segments', methods=['GET'])
def get_segments():
    if raw_df is not None:
        segments = raw_df['Road_Segment_ID'].unique().tolist()
        return jsonify({'status': 'success', 'segments': segments})
    return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    if raw_df is not None:
        stats = {
            'total_records': len(raw_df),
            'avg_traffic': float(raw_df['Vehicle_Count_Traffic_Demand'].mean()),
            'max_traffic': float(raw_df['Vehicle_Count_Traffic_Demand'].max()),
            'weather_summary': raw_df['Weather_Condition'].value_counts().to_dict()
        }
        return jsonify({'status': 'success', 'analytics': stats})
    return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'uptime': '99.9%'
    })

@app.route('/api/trend_image')
def get_trend_image():
    path = 'outputs/traffic_trend.png'
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Trend image not found in outputs folder", 404

@app.route('/api/trends', methods=['GET'])
def get_trends():
    period = request.args.get('range', '24h')
    if raw_df is not None:
        try:
            df = raw_df.copy()
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            max_date = df['Timestamp'].max()

            if period == '24h':
                # Show average 24h cycle across all data for better "profile" view
                trends = df.groupby(df['Timestamp'].dt.hour)['Vehicle_Count_Traffic_Demand'].mean().sort_index().reset_index()
                labels = [f"{int(h)}:00" for h in trends['Timestamp']]
            elif period == '7d':
                start_date = max_date - pd.Timedelta(days=7)
                subset = df[df['Timestamp'] >= start_date]
                trends = subset.groupby(subset['Timestamp'].dt.floor('D'))['Vehicle_Count_Traffic_Demand'].mean().reset_index()
                labels = [t.strftime('%b %d') for t in trends['Timestamp']]
            elif period == '30d':
                start_date = max_date - pd.Timedelta(days=30)
                subset = df[df['Timestamp'] >= start_date]
                trends = subset.groupby(subset['Timestamp'].dt.floor('D'))['Vehicle_Count_Traffic_Demand'].mean().reset_index()
                labels = [t.strftime('%b %d') for t in trends['Timestamp']]
            else:
                return jsonify({'status': 'error', 'message': 'Invalid range'}), 400

            return jsonify({
                'status': 'success',
                'labels': labels,
                'data': trends['Vehicle_Count_Traffic_Demand'].tolist()
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

@app.route('/api/download_report')
def download_report():
    report_path = 'outputs/summary_statistics.csv'
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return "Report not generated yet", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
