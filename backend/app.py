from flask import Flask, jsonify
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from routes.ai_routes import ai_bp

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Register Blueprint under /api
    app.register_blueprint(ai_bp, url_prefix="/api")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "backend-working"}), 200

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
