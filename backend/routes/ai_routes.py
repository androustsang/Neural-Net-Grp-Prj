
from flask import Blueprint, request, jsonify


# for Yehor and Maaz 
# from ml.model import run_classification
# from services.gemini_service import summarize_prediction

ai_bp = Blueprint("ai_routes", __name__)


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

  
    return jsonify(
        {
            "message": "Backend route is working. "
                       "CNN prediction is not wired yet. (Yehor's part)"
        }
    ), 501  # 501 = Not Implemented


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
