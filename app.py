from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__, template_folder="templates")
CORS(app)

# Load models (trained in older TF, works with 2.9.x/2.10.x)
plant_model = tf.keras.models.load_model("backend/plant_type_model1.h5", compile=False)
leaf_model = tf.keras.models.load_model("backend/plant_diseases_model.h5", compile=False)

# Upload folder for temporary storage
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check", methods=["POST"])
def check_leaf():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    # Predictions
    plant_pred = plant_model.predict(img_array)
    leaf_pred = leaf_model.predict(img_array)

    return jsonify({
        "plant_model_output": plant_pred.tolist(),
        "leaf_model_output": leaf_pred.tolist()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
