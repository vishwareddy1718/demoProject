from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("model.joblib")
encoder = joblib.load("encoder.joblib")
scaler = joblib.load("scaler.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

        # Arrange features in the same order used during training
        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        # Apply the same scaler used during training
        features_scaled = scaler.transform(features)

        # Predict encoded class
        prediction = model.predict(features_scaled)

        # Convert encoded class back to species name
        species = encoder.inverse_transform(prediction)[0]

        return jsonify({
            "success": True,
            "prediction": species
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)

