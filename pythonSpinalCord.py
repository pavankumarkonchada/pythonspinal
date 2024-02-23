from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import warnings
import requests
from sklearn.preprocessing import MinMaxScaler
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Enable CORS for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'OPTIONS, POST')
    return response

def download_model(url, local_path='model.pkl'):
    """Downloads the model file from the specified URL."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download model. Status code: {response.status_code}")
    return local_path

@app.route('/process_data', methods=['OPTIONS', 'POST'])
def process_data():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        received_data = data.get('data', [])
        df_t = pd.DataFrame(received_data)

        # Specify the URL to the trained model (use the raw content URL)
        model_url = "https://raw.githubusercontent.com/pavankumarkonchada/pythonspinal/main/xgboost_multilabel_model.pkl"
        
        # Download the model file
        model_path = download_model(model_url)

        # Load the trained model using pickle
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Convert categorical columns ('a', 'b', 'c') to numeric (0, 1, 2)
        category_to_number = {'a': 0, 'b': 1, 'c': 2}
        for column in df_t.columns:
            if set(df_t[column].unique()).issubset({'a', 'b', 'c'}):
                df_t[column] = df_t[column].map(category_to_number)

        # Normalize the input data
        scaler = MinMaxScaler()
        df_t_normalized = pd.DataFrame(scaler.fit_transform(df_t), columns=df_t.columns)

        # Make predictions
        prediction_result_df = pd.DataFrame(loaded_model.predict(df_t_normalized))

        # Convert predictions to a list
        updated_data = prediction_result_df.iloc[0].tolist()

        return jsonify({'message': 'Data processed successfully', 'updatedData': updated_data})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
