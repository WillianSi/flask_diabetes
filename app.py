import os
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle

app = Flask(__name__)

# Carrega o modelo treinado
model = pickle.load(open("modelo1.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        pred = model.predict(final_features)
        output = pred[0]

        if output == 0.0:
            text = "Você preenche os requisitos para não possuir diabetes, mesmo assim "
        elif output == 1.0:
            text = "Você tem tendência a possuir pré-diabetes, "
        else:
            text = "Você tem tendência a possuir diabetes, "

        return render_template("index.html", prediction_text="DIAGNÓSTICO: " + text + "procure um médico ou uma unidade de saúde.")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Ocorreu um erro: {e}")

@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])
    output = pred[0]
    return jsonify(output)

if __name__ == "__main__":
    # Para rodar localmente
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
