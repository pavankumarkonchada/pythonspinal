from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Function to download the model file
def download_model(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Download model at the start of the app to avoid repeated downloads
model_url = "https://raw.githubusercontent.com/pavankumarkonchada/pythonspinal/main/xgboost_multilabel_model.pkl"
model_path = "xgboost_multilabel_model.pkl"
download_model(model_url, model_path)

# Load the trained model using pickle
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Enable CORS for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'OPTIONS, POST')
    return response

@app.route('/process_data', methods=['OPTIONS', 'POST'])
def process_data():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        received_data = data.get('data', [])
        # Ensure received_data is a list of dictionaries for DataFrame creation
        if isinstance(received_data, list) and all(isinstance(item, dict) for item in received_data):
            df_t = pd.DataFrame(received_data)
        else:
            return jsonify({'error': 'Invalid data format. Expecting a list of dictionaries.'})

        # Convert categories 'a', 'b', 'c' to numeric values 0, 1, 2
        category_to_number = {'a': 0, 'b': 1, 'c': 2}
        for column in df_t.columns:
            if set(df_t[column].unique()).issubset({'a', 'b', 'c'}):
                df_t[column] = df_t[column].map(category_to_number)
        
        # Normalize the DataFrame
        scaler = MinMaxScaler()
        df_t_normalized = pd.DataFrame(scaler.fit_transform(df_t), columns=df_t.columns)
        
        # Make predictions
        prediction_result_df = pd.DataFrame(loaded_model.predict(df_t_normalized))
        updated_data = prediction_result_df.iloc[0].tolist()

        return jsonify({'message': 'Data processed successfully', 'updatedData': updated_data})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
