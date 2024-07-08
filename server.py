from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model from JSON
model = xgb.Booster()
model.load_model('model/fraud_detection_model.xg.model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Print the contents of the request
    # print("Received request data:", data)
    
    # Reorder the input data and split the category into individual binary characters
    input_data = {
        'index': data['id'],
        'Year': data['year'],
        'Month': data['month'],
        'Day': data['day'],
        'Hour': data['hour'],
        'Minute': data['minute'],
        'Amount': data['amount'],
        'Category_0': int(data['category'][0]),
        'Category_1': int(data['category'][1]),
        'Category_2': int(data['category'][2]),
        'Category_3': int(data['category'][3]),
        'Category_4': int(data['category'][4])
    }
    
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure the DataFrame columns match the expected feature names
    dmatrix = xgb.DMatrix(input_df)
    
    prediction = model.predict(dmatrix)
    is_fraud = prediction > 0.5
    return jsonify({'prediction':prediction.tolist(),'is_fraud': bool(is_fraud[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
