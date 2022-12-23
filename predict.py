from flask import Flask, request, jsonify
import pickle

# Creamos una instancia de Flask y cargamos el modelo entrenado en memoria
app = Flask(__name__)
with open("machismo_detector.pkl", "rb") as f:
    model = pickle.load(f)

# Creamos el endpoint que recibe una frase y devuelve si es machista o no
@app.route("/detect_machismo", methods=["POST"])
def detect_machismo():
    # Obtenemos la frase enviada en el cuerpo de la solicitud
    data = request.get_json()
    phrase = data["phrase"]

    # Procesamos la frase para convertirla en un vector de características
    vectorizer = CountVectorizer()
    X = vectorizer.transform([phrase])

    # Hacemos una predicción utilizando el modelo
    prediction = model.predict(X)[0]

    # Devolvemos la predicción en formato JSON
    return jsonify({"machista": prediction})

# Ejecutamos la aplicación de Flask
if __name__ == "__main__":
    app.run()
