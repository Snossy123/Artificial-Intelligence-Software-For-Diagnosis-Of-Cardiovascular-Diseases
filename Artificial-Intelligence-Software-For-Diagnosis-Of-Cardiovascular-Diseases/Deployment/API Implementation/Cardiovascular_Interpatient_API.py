import json
import pandas as pd 
from tensorflow.keras.models import load_model
from flask import Flask, request,jsonify
import numpy as np


app = Flask(name)

loaded_model = load_model("model.h5",compile = False)
out_col = ['output_CAD', 'output_CHF', 'output_MI', 
              'output_Normal']


def getLabel(file):
   pred = loaded_model.predict(file).round()
   output = {out_col[i]:pred[0][i] for i in range(4)}
   output_class = max(output, key=output.get).split('_')[-1]
   return output_class 
  
@app.route("/Check", methods=["POST"])
def create_page():
  file = request.files['messageFile']
  file.save("sample.csv")
  file = pd.read_csv("sample.csv")
  label = getLabel(file)
  return jsonify({'label':f'{label}'})

if name == "main":
  app.run("0.0.0.0",debug=True,port = 8000)