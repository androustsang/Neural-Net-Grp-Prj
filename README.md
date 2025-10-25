# Neural-Net-Grp-Prj
Neural Network Group Project (Group 6)  -  Developing Full-Stack AI-Enabled Applications

# Group 6

## Purpose and Vision

The goal of this project is to design, implement, and present a full-stack intelligent application that
addresses a real-world business or community problem expected to be relevant in 2026.
Your solution must demonstrate both deep learning with TensorFlow and Generative AI integration,
delivered through modern full-stack practices. Specifically, your system will
  1. Build and train a TensorFlow/Keras neural network (MLP, CNN, or Transformer) to perform prediction
or classification.
  2. Integrate a Generative AI component (Gemini API) for tasks such as summarization, explanation, or
chatbot interaction.
  3. Expose AI functionality via a backend API (Flask as default, FastAPI optional; REST required,
GraphQL optional).
  4. Develop a functional React 19 + Vite front-end to interact with the backend API and present results in
a user-friendly interface.
  5. Apply appropriate design patterns and principles (e.g., modular MVC structure in backend,
component-based architecture in frontend, API service abstraction) to ensure maintainability and clarity.

The end result should be a working prototype of an AI-enabled application that demonstrates both technical
depth and practical relevance to Canadian industry needs in 2026.

## Proposed Folder Structure (Draft only)

```
Neural-Net-Grp-Prj/
├── README.md
├── .gitignore
├── backend/                  # Flask API + TensorFlow + Gemini
│   ├── app.py                # Flask app entry point
│   ├── requirements.txt      # Backend dependencies
│   ├── config.py             # Configuration (API keys, paths)
│   ├── ml/
│   │   ├── model.py          # TensorFlow/Keras model definition
│   │   ├── train.py          # Training script
│   │   └── utils.py          # Helper functions (data loading, preprocessing)
│   ├── routes/
│   │   ├── ai_routes.py      # Endpoints for AI features (predict, summarize, chat)
│   │   └── __init__.py
│   ├── services/
│   │   ├── gemini_service.py # Wrapper for Gemini API calls
│   │   └── __init__.py
│   ├── static/               # Optional: serve static files if needed
│   └── templates/            # Optional: Jinja templates if using server-side pages
│
├── frontend/                 # React 19 + Vite frontend
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── PredictionForm.jsx
│   │   │   └── Chatbot.jsx
│   │   ├── pages/
│   │   │   ├── HomePage.jsx
│   │   │   ├── ResultsPage.jsx
│   │   └── services/
│   │       └── api.js        # Axios or Fetch abstraction for backend calls
│   └── public/
│       └── index.html
│
└── docs/                     # Documentation or design diagrams
    ├── architecture.md
    ├── dataflow-diagram.png
    └── api-spec.yaml
```