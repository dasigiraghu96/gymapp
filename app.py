# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:06:01 2019
@author: prithvi
"""
import flask
from flask import Flask, request,jsonify,render_template
import pickle
#from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np

app = Flask(__name__)
filename = 'Model_GYM.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
#@cross_origin
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
#@cross_origin
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)
