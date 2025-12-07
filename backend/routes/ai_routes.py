import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from flask import Blueprint, request, jsonify


from ml.model import CNNClassifier

# for Yehor and Maaz 
# from ml.model import run_classification
# from services.gemini_service import summarize_prediction

ai_bp = Blueprint("ai_routes", __name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../ml/models/binary_classifier_weighted.pth"
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded trained classifier from {MODEL_PATH}")
else:
    print(f"MODEL NOT FOUND at {MODEL_PATH}")
    
def preprocess_numpy(np_img):
    """
    Input: numpy array BGR (from cv2)
    Output: PyTorch tensor normalized & shaped like training data
    """
    img = cv2.resize(np_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

@ai_bp.route("/predict", methods=["POST"])
def predict():
    """
    Placeholder endpoint for running the classifier on an uploaded image.

    Expected final behaviour (after Yehor plugs in):
      - Method: POST
      - URL:    /api/predict
      - Body:   multipart/form-data with key "image" (file)
      - Output: JSON with prediction + confidence.

    For now, we only check the request and return a clear TODO message.
    """
    # 1) Check if an "image" file was sent in the request.
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file in form-data."}), 400

    file = request.files["image"]

    # 2) Check if filename is not empty.
    if not file or file.filename == "":
        return jsonify({"error": "Empty file."}), 400
    
    try:
        # Load image with OpenCV (your dataset uses cv2)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format."}), 400

        # Preprocess for model
        tensor = preprocess_numpy(img).to(device)

        # Inference
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits).item()
            
        print(prob)

        pothole = prob >= 0.8
        confidence = prob if pothole else (1 - prob)

        result = {
            "pothole_detected": pothole,
            "prediction": "pothole" if pothole else "no_pothole",
            "confidence": float(confidence),
            "raw_probability": float(prob)
        }

        main_path = "/Users/egor/Documents/GitHub"
        ddta_path = os.path.join(main_path, "Neural-Net-Grp-Prj/backend/data")
        predictions_dir = os.path.join(os.path.dirname(__file__), "/Users/egor/Documents/GitHub/Neural-Net-Grp-Prj/backend/data")
        os.makedirs(predictions_dir, exist_ok=True)

        # filename same as image but .txt
        base_name = os.path.splitext(file.filename)[0]
        save_path = os.path.join(predictions_dir, f"{base_name}.txt")

        with open(save_path, "w") as f:
            for key, value in result.items():
                f.write(f"{key}: {value}\n")

        print(f"Saved prediction to {save_path}")

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"Prediction Failed": str(e)}), 500


@ai_bp.route("/gen/summary", methods=["POST"])
def gen_summary():
    """
    Placeholder endpoint for generating a natural-language summary.

    Expected final behaviour (after Maaz plugs in):
      - Method: POST
      - URL:    /api/gen/summary
      - Body:   JSON with keys:
          { "prediction": "...", "confidence": 0.91, "extraContext": "..." }
      - Output: JSON with key:
          { "summary": "..." }

    For now, we validate input and return a clear TODO message.
    """
    data = request.get_json(silent=True) or {}

    prediction = data.get("prediction")
    confidence = data.get("confidence")
    extra_context = data.get("extraContext", "")

    # Basic validation on prediction
    if prediction not in ("pothole", "no_pothole"):
        return jsonify({"error": "Invalid or missing 'prediction'."}), 400

    # Basic validation on confidence
    try:
        float(confidence)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing 'confidence'."}), 400

    # Final behaviour (later):
    #   summary_text = summarize_prediction(prediction, float(confidence), extra_context)
    #   return jsonify({"summary": summary_text}), 200

    return jsonify(
        {
            "message": "Backend route is working. "
                       "Gemini summary is not wired yet. (Maaz's part)"
        }
    ), 501
