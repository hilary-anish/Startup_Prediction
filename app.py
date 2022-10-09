import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template


# defining a flask application

app = Flask(__name__)

# Loading the model
xgb_model = pickle.load(open('xgboost.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = xgb_model.predict(new_data)
    print(f'the output is {output}')
    return jsonify(int(output[0]))


app.run(debug=True)