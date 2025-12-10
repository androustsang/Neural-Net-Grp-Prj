# Maaz Bobat, Saaram Rashidi, MD Sazid, Sun Hung Tsang, Yehor Valesiuk
from flask import Flask, request, jsonify
from flask_cors import CORS
from gemini_service import get_rag_graph

app = Flask(__name__)
CORS(app)

# Initialize the RAG pipeline once
graph = get_rag_graph()
qa_graph = graph.compile()


@app.route("/rag/chat", methods=["POST"])
def chat():
    """
    Endpoint to receive a question and invoke the RAG LangGraph workflow.
    Matches assignment spec: POST /api/chat
    """
    data = request.get_json()

    question = data.get("message", "")
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "Missing message"}), 400

    try:
        # Initial state for the LangGraph
        # include history for short‑term memory
        state = {
            "query": question,
            "history": history,  # pass chat history into graph
        }

        # Invoke the compiled graph (Retrieval -> Generation)
        result = qa_graph.invoke(state)

        answer = result.get("answer", "")

        # The 'context' key holds the raw text chunks retrieved from FAISS
        retrieved_context = result.get("context", "No context retrieved.")

        updated_history = result.get("history", history)

        # Extract and combine sources (Local + Web)
        source_docs = result.get("source_documents", [])
        local_sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]

        web_sources_list = result.get("web_sources", [])
        web_sources = [s.get("url", "Unknown Web Source") for s in web_sources_list]

        combined_sources = list(set(local_sources + web_sources))

        # Return the structured response
        return jsonify(
            {
                "Model response": answer,  # "Model response"
                "Retrieved context": retrieved_context,  # "Retrieved context"
                "sources": combined_sources,  # Helpful extra metadata like sources
                "history": updated_history,  # pass back history
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
