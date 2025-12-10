# Maaz Bobat, Saaram Rashidi, MD Sazid, Sun Hung Tsang, Yehor Valesiuk

import cv2  # Needed for image processing and it needs to be first line to avoid conflicts
from flask import Flask, jsonify
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from routes.ai_routes import ai_bp

import torch  # Needed for loading CNN and YOLO models
from ultralytics import YOLO  # For YOLO model loading - object detection


def load_models():
    """Helper function to load models so create_app isn't too messy"""
    print("Loading AI Models...")

    # 1. Load YOLO
    # Ensure you point to the correct path of your .pt file
    yolo_model = YOLO("ml/models/yolo_v12small_best.pt")

    # 2. Load CNN (Example logic)
    # cnn_model = MyCustomCNN()
    # cnn_model.load_state_dict(torch.load("ml/models/cnn_weights.pth", map_location='cpu'))
    # cnn_model.eval()

    print("Models loaded successfully.")

    # Return a dictionary containing both
    return {
        "yolo": yolo_model,
        "cnn": None,  # Replace with cnn_model variable when you have it
    }


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # --- THE MAGICAL STEP ---
    # Load models and attach them to app.extensions
    # This makes them accessible everywhere in the project
    app.extensions["ml_models"] = load_models()

    # Register Blueprint under /api
    app.register_blueprint(ai_bp, url_prefix="/api")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "backend-working"}), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
