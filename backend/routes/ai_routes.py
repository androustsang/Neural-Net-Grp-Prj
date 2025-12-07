from flask import Blueprint, request, jsonify, current_app # To access app context (current_app)
from services.yolo_service import process_yolo_prediction # To process YOLO predictions


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

    try:
        # 1. Get the model from the global app context
        # (This was set up in app.py in the previous step)
        yolo_model = current_app.extensions["ml_models"]["yolo"]

        # 2. Pass the model and the file to our custom service
        # This function handles the "pass", counting, and image drawing
        prediction_result = process_yolo_prediction(yolo_model, file)

        # 3. Return the clean JSON
        return (
            jsonify(
                {
                    "message": "Prediction successful",
                    "pothole_count": prediction_result["count"],
                    "annotated_image": prediction_result[
                        "image_data"
                    ],  # The image is here!
                    "cnn_result": "Pending implementation",
                }
            ),
            200,
        )

    except Exception as e:
        # Good practice: Print the error to your console so you can debug
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Internal processing error"}), 500

    # return jsonify(
    #     {
    #         "message": "Backend route is working. "
    #                    "CNN prediction is not wired yet. (Yehor's part)"
    #     }
    # ), 501  # 501 = Not Implemented


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

    return (
        jsonify(
            {
                "message": "Backend route is working. "
                "Gemini summary is not wired yet. (Maaz's part)"
            }
        ),
        501,
    )
