import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("modelo1.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)
    output = pred[0]

    if(output == 0.0):
        text = "Você preenche os requisitos para não possuir diabetes, mesmo assim "
    elif(output == 1.0):
        text = "Você tem tendência a possuir pré-diabetes, "
    else:
        text = "Você tem tendência a possuir diabetes, "

    return render_template("index.html", prediction_text="DIAGNOSTICO: " + text + "procure um medico ou uma unidade de saude.")


@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    output = pred[0]
    return jsonify(output)
