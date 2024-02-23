from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from scipy.stats.mstats import winsorize
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

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
        df_t = pd.DataFrame(received_data, index=[0])
        print(df_t)
  

    # Specify the path to the trained model
        model_path = "E:/Projects/IBS/Model/xgboost_multilabel_model.pkl"

    # Load the trained model using pickle
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        
    

        category_to_number = {'a': 0, 'b': 1, 'c': 2}

# Check each column in x_train for the presence of 'a', 'b', 'c'
        for column in df_t.columns:
    # Check if the column contains any of the specified categories
            if set(df_t[column].unique()).issubset({'a', 'b', 'c'}):
        # Apply the mapping
                df_t[column] = df_t[column].map(category_to_number)
                scaler = MinMaxScaler()
                df_t_normalized = pd.DataFrame(scaler.fit_transform(df_t),columns=df_t.columns)
    # Reorder columns in df2 to match the order of df1
            #df_t = df_t[x_train.columns]
        
        prediction_result_df = pd.DataFrame(loaded_model.predict(df_t_normalized))

# Extract the values of the first row and convert to a Python list
        updated_data = prediction_result_df.iloc[0].tolist()


        return jsonify({'message': 'Data processed successfully', 'updatedData': updated_data})
    except Exception as e:
        return jsonify({'error':str(e)})

if __name__ == '__main__':
    app.run(debug=True)
