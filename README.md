# Neural-Net-Grp-Prj
**Neural Network Group Project (Group 6)** - Developing Full-Stack AI-Enabled Applications for Pothole Detection and Reporting

## 🎯 Purpose and Vision

This project delivers a full-stack intelligent application that addresses real-world infrastructure maintenance challenges expected to be relevant in 2026. Our solution combines **computer vision for pothole detection** with **AI-powered assistance** for reporting and information retrieval.

### Core Capabilities

1. **Deep Learning Vision System**: YOLOv12-based CNN for real-time pothole detection from images
2. **Generative AI Integration**: Gemini API-powered RAG (Retrieval-Augmented Generation) chatbot for:
   - Answering questions about pothole causes, fixing procedures, and reporting processes
   - Providing context-aware responses using local knowledge base
   - Fallback to Google Search for information beyond local documents
3. **RESTful Backend API**: Flask-based service with modular architecture
4. **Modern Frontend**: React 19 + Vite dashboard for image upload, detection visualization, and interactive chat
5. **Production-Ready Design**: Component-based architecture, API abstraction, and maintainable codebase

---

## 📂 Project Structure

```
Neural-Net-Grp-Prj/
├── README.md                          # Project overview and documentation
├── requirements.txt                   # Python dependencies (backend)
│
├── backend/                           # Flask API + YOLO + Gemini
│   ├── .gitignore                     # Backend-specific ignore rules
│   ├── app.py                         # Main Flask application entry point
│   ├── config.py                      # Configuration management (API keys, paths, env vars)
│   │
│   ├── data/                          # Knowledge base for RAG system
│   │   ├── cause_of_potholes.txt      # Educational content about pothole formation
│   │   ├── fixing_potholes.txt        # Repair procedures and best practices
│   │   ├── pothole_request.txt        # How to submit pothole reports
│   │   ├── report_pothole.txt         # Reporting guidelines
│   │   ├── potholes_data.txt          # General pothole information
│   │   ├── email_template.txt         # Template for automated communications
│   │   ├── predictions_log.txt        # Log file for detection history
│   │   ├── README.dataset.txt         # Dataset documentation (Roboflow)
│   │   └── README.roboflow.txt        # Roboflow integration notes
│   │
│   ├── ml/                            # Machine Learning module (legacy/experimental)
│   │   ├── .gitkeep                   # Placeholder for ML experiments
│   │   └── [training scripts]         # Various training and testing files
│   │
│   ├── routes/                        # API endpoints
│   │   ├── ai_routes.py               # Core AI endpoints:
│   │   │                              #   - POST /api/predict (YOLO detection)
│   │   │                              #   - POST /api/gen/summary (Gemini summary)
│   │   │                              #   - GET /api/health (Health check)
│   │   └── __init__.py
│   │
│   ├── services/                      # Business logic layer
│   │   ├── gemini_service.py          # RAG system with LangGraph workflow:
│   │   │                              #   - Vector store (FAISS)
│   │   │                              #   - Agentic routing (retrieve/generate)
│   │   │                              #   - Memory management
│   │   │                              #   - Google Search grounding
│   │   ├── yolo_service.py            # YOLO model wrapper:
│   │   │                              #   - Model loading
│   │   │                              #   - Image preprocessing
│   │   │                              #   - Bounding box annotation
│   │   │                              #   - Base64 encoding for web
│   │   ├── app.py                     # Standalone RAG chat API (port 5000):
│   │   │                              #   - POST /api/chat
│   │   └── __init__.py
│   │
│   ├── static/                        # Static assets (if needed)
│   └── __pycache__/                   # Python bytecode cache
│
├── dashboard/                         # React 19 + Vite frontend
│   ├── .gitignore                     # Frontend-specific ignore rules
│   ├── package.json                   # Node.js dependencies and scripts
│   ├── vite.config.js                 # Vite build configuration
│   ├── eslint.config.js               # ESLint linting rules
│   ├── index.html                     # Entry HTML file
│   ├── README.md                      # Frontend-specific documentation
│   │
│   ├── public/                        # Static public assets
│   │   └── [images, icons, etc.]
│   │
│   └── src/                           # React source code
│       ├── App.jsx                    # Main app component with routing
│       │
│       ├── components/                # Reusable React components
│       │   ├── Navbar.jsx             # Navigation bar
│       │   ├── PredictionForm.jsx     # Image upload form
│       │   ├── ResultsDisplay.jsx     # Detection results visualization
│       │   └── RagChat.jsx            # Interactive chatbot interface
│       │
│       ├── pages/                     # Page-level components
│       │   ├── PotholeDetector.jsx    # Main detection page
│       │   └── [other pages]
│       │
│       └── services/                  # API abstraction layer
│           └── api.js                 # Axios/Fetch wrapper for backend calls
│
├── docs/                              # Documentation
│   └── .gitkeep                       # Placeholder
│
└── public/                            # Shared public assets
    ├── .gitkeep                       # Placeholder
    └── demo/                          # Demo materials and screenshots
```

---

## 🏗️ Architecture Overview

### Backend Components

#### **1. Flask Application ([`backend/app.py`](backend/app.py))**
- Initializes Flask app with CORS support
- Registers blueprint routes under `/api` prefix
- Health check endpoint for monitoring

#### **2. AI Routes ([`backend/routes/ai_routes.py`](backend/routes/ai_routes.py))**
- **`POST /api/predict`**: Accepts image uploads, runs YOLO detection, returns annotated image with pothole count
- **`POST /api/gen/summary`**: Generates natural language summaries of detection results using Gemini
- **`GET /api/health`**: Service health status

#### **3. YOLO Service ([`backend/services/yolo_service.py`](backend/services/yolo_service.py))**
- Loads pre-trained YOLOv12 model for pothole detection
- Processes uploaded images with `supervision` library for annotation
- Returns bounding boxes, confidence scores, and base64-encoded annotated images

#### **4. Gemini RAG Service ([`backend/services/gemini_service.py`](backend/services/gemini_service.py))**
- **Vector Store**: FAISS embeddings of local documents in `backend/data/`
- **LangGraph Workflow**: 4-node agentic system
  1. **Router Node**: Determines if document retrieval is needed
  2. **Retrieve Node**: Fetches relevant context from vector store
  3. **Generate Node**: Uses Gemini 2.0 Flash with Google Search grounding
  4. **Memory Node**: Maintains conversation history
- **Standalone API** ([`backend/services/app.py`](backend/services/app.py)): Separate chat server on port 5000

#### **5. Knowledge Base ([`backend/data/`](backend/data/))**
Text files containing domain knowledge:
- Pothole causes and formation
- Repair methodologies
- Reporting procedures
- Email templates for notifications

---

### Frontend Components

#### **1. Main App ([`dashboard/src/App.jsx`](dashboard/src/App.jsx))**
- React Router setup with routes:
  - `/` → Pothole Detector page
  - `/ragchat` → RAG Chat interface

#### **2. Pothole Detector Page ([`dashboard/src/pages/PotholeDetector.jsx`](dashboard/src/pages/PotholeDetector.jsx))**
- Image upload interface
- Calls `/api/predict` endpoint
- Displays annotated images with detection results
- Shows pothole count and confidence scores

#### **3. RAG Chat Component ([`dashboard/src/components/RagChat.jsx`](dashboard/src/components/RagChat.jsx))**
- Interactive chatbot UI with message history
- Sends queries to `/api/chat` (standalone RAG service)
- Displays:
  - AI-generated responses
  - Retrieved document context (collapsible)
  - Web sources with clickable links
  - Conversation history for context

#### **4. API Service Layer ([`dashboard/src/services/api.js`](dashboard/src/services/api.js))**
- Abstraction layer for backend communication
- Functions: `predict()`, `generateSummary()`, `chatWithAI()`
- Currently uses mock implementations (ready for integration)

---

## 🚀 Getting Started

### Prerequisites
- **Backend**: Python 3.10+, Flask, TensorFlow, Ultralytics YOLO, LangChain, FAISS
- **Frontend**: Node.js 18+, npm/yarn
- **API Keys**: Google Gemini API key (set in environment variables)

### Installation

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY="your_gemini_api_key"
export FLASK_APP=app.py

# Run main Flask server
python -m flask run
# Server runs on http://localhost:5000

# (Optional) Run standalone RAG chat server
cd services
python app.py
# RAG server runs on http://localhost:5000
```

#### Frontend Setup
```bash
cd dashboard
npm install
npm run dev
# Development server runs on http://localhost:5173
```

---

## 🔌 API Endpoints

### Main Flask API (`http://localhost:5000`)

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/health` | GET | Health check | None | `{"status": "ok"}` |
| `/api/predict` | POST | YOLO pothole detection | `multipart/form-data` (image) | `{"count": int, "image_data": "base64..."}` |
| `/api/gen/summary` | POST | Generate detection summary | `{"prediction": str, "confidence": float}` | `{"summary": "text..."}` |

### RAG Chat API (`http://localhost:5000` - separate service)

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/chat` | POST | Agentic RAG chatbot | `{"message": str, "history": []}` | `{"Model response": str, "Retrieved context": str, "sources": [], "history": []}` |

---

## 🛠️ Technologies Used

### Backend
- **Framework**: Flask (REST API)
- **Deep Learning**: Ultralytics YOLOv8, TensorFlow/Keras
- **Generative AI**: Google Gemini 2.0 Flash API
- **RAG Stack**: LangChain, LangGraph, FAISS (vector store)
- **Computer Vision**: OpenCV, Supervision (bounding box annotation)
- **Data Processing**: NumPy, Pandas

### Frontend
- **Framework**: React 19
- **Build Tool**: Vite
- **Styling**: CSS Modules / Tailwind (TBD)
- **Routing**: React Router v6
- **HTTP Client**: Axios / Fetch API

### DevOps
- **Version Control**: Git, GitHub
- **Environment Management**: Conda (Python), npm (Node.js)
- **API Testing**: Postman / Thunder Client

---

## 📊 RAG Workflow Diagram

```
User Query
    ↓
[Router Node]  ────→ (Needs retrieval?)
    ↓ Yes                    ↓ No
[Retrieve Node]              ↓
    ↓                        ↓
(FAISS Vector Store)         ↓
    ↓                        ↓
[Generate Node] ←────────────┘
    ↓
(Gemini 2.0 + Google Search)
    ↓
[Memory Node]
    ↓
Response (Answer + Sources + History)
```

---

## 🎓 Project Context

- **Course**: Neural Network Applications
- **Team**: Group 6
- **Objective**: Full-stack AI application for 2026 Canadian industry needs
- **Use Case**: Municipal infrastructure maintenance and citizen reporting

---

## 📝 Development Notes

### Current Status
- ✅ **YOLO Detection**: Fully implemented with YOLOv12
- ✅ **RAG Chatbot**: Functional with LangGraph workflow
- ✅ **Frontend UI**: React dashboard with detection and chat interfaces
- ⚠️ **Integration**: Frontend API calls currently use mocks (ready for backend integration)
- 🔄 **In Progress**: Gemini summary generation, predictions logging

### Known Issues
- OpenCV DLL loading on Windows (see troubleshooting in project wiki)
- CORS configuration may need adjustment for production deployment
- Vector store rebuilds on each RAG service startup (consider persistence)

### Future Enhancements
- User authentication and session management
- Database for predictions logging (PostgreSQL/MongoDB)
- Real-time notifications for pothole reports
- Mobile-responsive design optimization
- Deployment to cloud platform (AWS/GCP/Azure)

---

## 👥 Contributors

**Group 6 Members**:


---

## 📄 License

[Specify license - MIT, Apache 2.0, etc.]

---

## 🔗 References

- [Ultralytics YOLOv12 Documentation](https://docs.ultralytics.com/)
- [Google Gemini API](https://ai.google.dev/)
- [LangChain Documentation](https://python.langchain.com/)
- [React 19 Documentation](https://react.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)