# flask web app for image classification

import os
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

# flask setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# configuration
IMG_SIZE = 150
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_DIR = "models"
UPLOAD_DIR = os.path.join("static", "uploads")

# class descriptions
CLASS_DESCRIPTIONS = {
    "buildings": "Urban Buildings - City skylines, houses, or structures",
    "forest": "Forest - Dense tree coverage, woodlands, jungles",
    "glacier": "Glacier - Ice formations, snowy or icy landscapes",
    "mountain": "Mountain - Peaks, hills, rocky elevated terrain",
    "sea": "Sea - Ocean, water bodies, coastlines",
    "street": "Street - Roads, paths, urban ground-level views"
}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# preprocessing pytorch
def preprocess_image_for_pytorch(image: Image.Image):
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor

# preprocessing tensorflow
def preprocess_image_for_tensorflow(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# prediction pytorch
def predict_pytorch(image: Image.Image):
    import torch

    model_path = os.path.join(MODEL_DIR, "Tikome_Nana_model.pth")
    if not os.path.exists(model_path):
        return None, None, None, "PyTorch model not found. Please train first"

    import torch.nn as nn

    class IntelCNN_PyTorch(nn.Module):
        def __init__(self, num_classes=6):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(128 * 18 * 18, 512),
                nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelCNN_PyTorch(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tensor = preprocess_image_for_pytorch(image).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = probs.max(0)

    class_name = CLASSES[pred_idx.item()]
    confidence = conf.item() * 100
    all_probs = {CLASSES[i]: round(probs[i].item() * 100, 2) for i in range(6)}

    return class_name, confidence, all_probs, None

# prediction tensorflow
def predict_tensorflow(image: Image.Image):
    import tensorflow as tf

    model_path = os.path.join(MODEL_DIR, "Tikome_Nana_model.keras")
    if not os.path.exists(model_path):
        return None, None, None, "TensorFlow model not found. Please train first"

    model = tf.keras.models.load_model(model_path)
    arr = preprocess_image_for_tensorflow(image)

    preds = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(preds)
    class_name = CLASSES[pred_idx]
    confidence = float(preds[pred_idx]) * 100
    all_probs = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(6)}

    return class_name, confidence, all_probs, None

# routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form.get("model", "pytorch")
    if model_choice not in ["pytorch", "tensorflow"]:
        return jsonify({"error": "Invalid model choice"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_ext = {"jpg", "jpeg", "png", "bmp", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"File type .{ext} not supported"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    try:
        if model_choice == "pytorch":
            class_name, confidence, all_probs, error = predict_pytorch(image)
        else:
            class_name, confidence, all_probs, error = predict_tensorflow(image)

        if error:
            return jsonify({"error": error}), 503

        return jsonify({
            "predicted_class": class_name,
            "confidence": round(confidence, 2),
            "all_probs": all_probs,
            "description": CLASS_DESCRIPTIONS.get(class_name, ""),
            "model_used": model_choice
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/health")
def health():
    pt_model = os.path.exists(os.path.join(MODEL_DIR, "Tikome_Nana_model.pth"))
    tf_model = os.path.exists(os.path.join(MODEL_DIR, "Tikome_Nana_model.keras"))
    return jsonify({
        "status": "ok",
        "pytorch_model": "loaded" if pt_model else "not found",
        "tensorflow_model": "loaded" if tf_model else "not found"
    })

# run server
if __name__ == "__main__":
    print("Starting Intel Image Classifier Web App")
    print("Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
