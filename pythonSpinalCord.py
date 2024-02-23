from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import warnings
import requests
warnings.filterwarnings('ignore')
import xgboost as xgb

app = Flask(__name__)

def download_model(url, local_filename='model.pkl'):
    """Download a file from a URL to a local file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# URL to the .pkl model file in GitHub (use the raw content URL)
model_url = "https://github.com/pavankumarkonchada/pythonspinal/raw/main/xgboost_multilabel_model.pkl"

# Download the model file at app startup
model_path = download_model(model_url)

# Load the trained model using pickle
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        received_data = data.get('data', {})
        df_t = pd.DataFrame(received_data, index=[0])

        category_to_number = {'a': 0, 'b': 1, 'c': 2}
        for column in df_t.columns:
            if df_t[column].dtype == 'object' and set(df_t[column].unique()).issubset({'a', 'b', 'c'}):
                df_t[column] = df_t[column].map(category_to_number)

        scaler = MinMaxScaler()
        df_t_normalized = pd.DataFrame(scaler.fit_transform(df_t), columns=df_t.columns)

        prediction = loaded_model.predict(df_t_normalized)
        updated_data = prediction.tolist()

        return jsonify({'message': 'Data processed successfully', 'prediction': updated_data})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
