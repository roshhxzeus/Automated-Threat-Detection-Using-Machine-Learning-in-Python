from flask import Flask, request, jsonify, render_template_string
import numpy as np
import traceback
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import cudf
from cuml.ensemble import RandomForestClassifier
import pickle
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, filename='atd.log', format='%(asctime)s - %(message)s')

DATASET_FILES = [
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv'
]

FEATURE_COLS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
    'Flow Packets/s', 'Fwd PSH Flags', 'Bwd PSH Flags', 'SYN Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'Packet Length Mean'
]

# Global list to store prediction history
prediction_history = []

def preprocess_and_merge_data(data_dir):
    df_list = []
    for file in DATASET_FILES:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            logging.warning(f"{file_path} not found, skipping...")
            continue
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df_list.append(df)
    
    if not df_list:
        raise ValueError("No valid dataset files found.")
    
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    merged_df['Label'] = merged_df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    print(f"Raw label distribution: {np.bincount(merged_df['Label'])}")
    
    X = merged_df[FEATURE_COLS].astype(np.float32)
    y = merged_df['Label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def load_or_train_models(data_dir):
    if os.path.exists('iso_forest.pkl') and os.path.exists('rf_model.pkl') and os.path.exists('scaler.pkl'):
        with open('iso_forest.pkl', 'rb') as f:
            iso_forest = pickle.load(f)
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        X, y, scaler = preprocess_and_merge_data(data_dir)
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        iso_labels = iso_forest.fit_predict(X)
        iso_labels = np.where(iso_labels == -1, 0, 1)
        
        smote = SMOTE(random_state=42, n_jobs=-1)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        X_balanced = X_balanced.astype(np.float32)
        print(f"Balanced label distribution: {np.bincount(y_balanced)}")
        
        X_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_balanced))
        y_gpu = cudf.Series(y_balanced)
        
        X_train, X_test, y_train, y_test = train_test_split(X_gpu, y_gpu, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(
            n_estimators=2000, 
            max_depth=150, 
            min_samples_split=5, 
            random_state=42, 
            n_streams=1
        )
        print("Training Random Forest on GPU...")
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test.to_numpy(), rf_pred.to_numpy())
        pred_dist = np.bincount(rf_pred.to_numpy().astype(int))
        print(f"Random Forest Accuracy: {accuracy}")
        print(f"Predicted label distribution on test set: {pred_dist}")
        logging.info(f"Random Forest Accuracy: {accuracy}")
        logging.info(f"Predicted label distribution on test set: {pred_dist}")
        
        with open('iso_forest.pkl', 'wb') as f:
            pickle.dump(iso_forest, f)
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    return iso_forest, rf_model, scaler

DATA_DIR = '/mnt/c/atd/DATADIR'
iso_forest, rf_model, scaler = load_or_train_models(DATA_DIR)

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Anomaly Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Orbitron', 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 5px;
            background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3d 100%);
            color: #fff;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        h1 {
            text-align: center;
            color: #00d4ff;
            text-shadow: 0 0 15px rgba(0, 212, 255, 0.7);
            font-size: 1.8em;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }
        .container {
            display: flex;
            flex: 1;
            gap: 10px;
            max-width: 90%;
        }
        .left-panel {
            width: 40%; /* Wider for charts */
            max-width: 300px; /* Increased cap */
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .right-panel {
            flex-grow: 1; /* Fill remaining space */
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px; /* Slightly larger gap */
        }
        label {
            font-weight: 600;
            color: #b0b0ff;
            margin-bottom: 4px;
            display: block;
            text-transform: uppercase;
            font-size: 0.8em;
        }
        input[type="number"] {
            width: 150px; /* Larger inputs */
            padding: 8px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: #00d4ff;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            background: rgba(255, 255, 255, 0.15);
        }
        button {
            grid-column: span 3;
            padding: 12px;
            background: linear-gradient(90deg, #00d4ff 0%, #007bff 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
        }
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.6);
            background: linear-gradient(90deg, #00e4ff, #0088ff);
        }
        #result {
            margin-top: 10px;
            padding: 10px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            text-align: center;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
        }
        .error {
            color: #ff5555;
            font-weight: bold;
        }
        .benign, .normal { color: #55ff55; }
        .malicious, .anomaly { color: #ff5555; }
        canvas {
            width: 200px !important; /* Adjusted chart size */
            height: 200px !important;
            max-width: 200px;
            max-height: 200px;
            aspect-ratio: 1;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <canvas id="confidenceChart"></canvas>
            <canvas id="historyChart"></canvas>
        </div>
        <div class="right-panel">
            <h1>Network Anomaly Detection</h1>
            <form id="predictForm">
                <div class="form-grid">
                    <div>
                        <label>Flow Duration</label>
                        <input type="number" name="Flow Duration" required step="1" min="0" value="0">
                    </div>
                    <div>
                        <label>Fwd Packets</label>
                        <input type="number" name="Total Fwd Packets" required step="1" min="0" value="0">
                    </div>
                    <div>
                        <label>Bwd Packets</label>
                        <input type="number" name="Total Backward Packets" required step="1" min="0" value="0">
                    </div>
                    <div>
                        <label>Fwd Pkt Len</label>
                        <input type="number" name="Fwd Packet Length Mean" required step="0.01" min="0" value="0">
                    </div>
                    <div>
                        <label>Bwd Pkt Len</label>
                        <input type="number" name="Bwd Packet Length Mean" required step="0.01" min="0" value="0">
                    </div>
                    <div>
                        <label>Flow Bytes/s</label>
                        <input type="number" name="Flow Bytes/s" required step="0.01" value="0">
                    </div>
                    <div>
                        <label>Flow Pkts/s</label>
                        <input type="number" name="Flow Packets/s" required step="0.01" value="0">
                    </div>
                    <div>
                        <label>Fwd PSH</label>
                        <input type="number" name="Fwd PSH Flags" required step="1" min="0" max="1" value="0">
                    </div>
                    <div>
                        <label>Bwd PSH</label>
                        <input type="number" name="Bwd PSH Flags" required step="1" min="0" max="1" value="0">
                    </div>
                    <div>
                        <label>SYN Count</label>
                        <input type="number" name="SYN Flag Count" required step="1" min="0" value="0">
                    </div>
                    <div>
                        <label>ACK Count</label>
                        <input type="number" name="ACK Flag Count" required step="1" min="0" value="0">
                    </div>
                    <div>
                        <label>URG Count</label>
                        <input type="number" name="URG Flag Count" required step="1" min="0" value="0">
                    </div>
                    <div>
                        <label>Pkt Len Mean</label>
                        <input type="number" name="Packet Length Mean" required step="0.01" min="0" value="0">
                    </div>
                </div>
                <button type="submit">Predict</button>
            </form>
            <div id="result" style="display: none;">
                <p id="textResult"></p>
            </div>
        </div>
    </div>

    <script>
        let confidenceChart, historyChart;
        let historyData = { normal: 0, anomaly: 0 };

        document.getElementById('predictForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                const resultDiv = document.getElementById('result');
                const textResult = document.getElementById('textResult');
                
                if (result.error) {
                    textResult.innerHTML = `<p class="error">Error: ${result.error}</p>`;
                    resultDiv.style.display = 'block';
                    destroyCharts();
                } else {
                    textResult.innerHTML = `
                        <p><strong>Isolation Forest:</strong> <span class="${result.isolation_forest_result}">${result.isolation_forest_result}</span></p>
                        <p><strong>Random Forest:</strong> <span class="${result.random_forest_result}">${result.random_forest_result}</span></p>
                    `;
                    resultDiv.style.display = 'block';
                    updateCharts(result);
                    updateHistory(result.random_forest_result);
                }
            })
            .catch(error => {
                document.getElementById('textResult').innerHTML = `<p class="error">Error: ${error.message}</p>`;
                document.getElementById('result').style.display = 'block';
                destroyCharts();
            });
        });

        function destroyCharts() {
            if (confidenceChart) confidenceChart.destroy();
            if (historyChart) historyChart.destroy();
        }

        function updateCharts(result) {
            destroyCharts();

            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            confidenceChart = new Chart(confidenceCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Confidence', 'Remaining'],
                    datasets: [{
                        data: [result.confidence * 100, (1 - result.confidence) * 100],
                        backgroundColor: ['#00d4ff', '#555'],
                        borderWidth: 0
                    }]
                },
                options: {
                    circumference: 180,
                    rotation: -90,
                    cutout: '70%',
                    aspectRatio: 1,
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false },
                        title: {
                            display: true,
                            text: `Confidence: ${(result.confidence * 100).toFixed(2)}%`,
                            color: '#fff',
                            font: { size: 12 }
                        }
                    }
                }
            });

            const historyCtx = document.getElementById('historyChart').getContext('2d');
            historyChart = new Chart(historyCtx, {
                type: 'pie',
                data: {
                    labels: ['Normal', 'Anomaly'],
                    datasets: [{
                        data: [historyData.normal, historyData.anomaly],
                        backgroundColor: ['#55ff55', '#ff5555'],
                        borderWidth: 1,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    plugins: {
                        title: {
                            display: true,
                            text: 'Prediction History',
                            color: '#fff',
                            font: { size: 12 }
                        }
                    }
                }
            });
        }

        function updateHistory(rf_result) {
            if (rf_result === 'normal') {
                historyData.normal++;
            } else {
                historyData.anomaly++;
            }
            if (historyChart) {
                historyChart.data.datasets[0].data = [historyData.normal, historyData.anomaly];
                historyChart.update();
            }
        }
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        missing = [col for col in FEATURE_COLS if col not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400
        
        try:
            for col in FEATURE_COLS:
                data[col] = float(data[col])
        except ValueError as e:
            return jsonify({'error': f'Invalid numeric value: {str(e)}'}), 400
        
        for col in ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Packet Length Mean']:
            if data[col] < 0:
                return jsonify({'error': f'Negative value not allowed for {col}'}), 400
        
        features_df = pd.DataFrame([data], columns=FEATURE_COLS).astype(np.float32)
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
        if features_df.empty:
            return jsonify({'error': 'Invalid feature data after processing'}), 400
        
        features_scaled = scaler.transform(features_df)
        iso_pred = iso_forest.predict(features_scaled)
        iso_result = 'malicious' if iso_pred[0] == -1 else 'benign'
        
        features_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(features_scaled, columns=FEATURE_COLS))
        rf_proba = rf_model.predict_proba(features_gpu)
        rf_pred_np = 1 if rf_proba.iloc[0, 1] > 0.06 else 0  # Lowered threshold to 0.05
        rf_result = 'anomaly' if rf_pred_np == 1 else 'normal'
        
        confidence = float(rf_proba.iloc[0, 1])  # Probability of anomaly (class 1)
        logging.info(f"Raw features: {data}")
        logging.info(f"Scaled features: {features_scaled}")
        logging.info(f"RF Proba: {rf_proba.to_string()}, Prediction: {rf_pred_np}, Result: iso={iso_result}, rf={rf_result}, confidence={confidence}")
        
        # Update prediction history
        prediction_history.append(rf_result)
        
        return jsonify({
            'isolation_forest_result': iso_result,
            'random_forest_result': rf_result,
            'confidence': confidence
        })
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
