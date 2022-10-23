import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)
    output = pred[0]
    p = output

    if(p == 0):
        text = "Você preenche os requisitos para não possuir diabetes, mesmo assim "
    elif(p == 1):
        text = "Você tem tendência a possuir pré-diabetes, "
    else:
        text = "Você tem tendência a possuir diabetes, "

    return render_template("index.html", prediction_text="DIAGNÓSTICO: " + text + "procure um médico ou uma unidade de saúde.")

@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    output = pred[0]
    return jsonify(output)
