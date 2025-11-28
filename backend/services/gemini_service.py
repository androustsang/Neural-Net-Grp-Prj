import os
from typing import TypedDict, List
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# SETUP & CONFIGURATION
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")
client = genai.Client(api_key=GEMINI_API_KEY)


# VECTOR STORE SETUP WITH FALLBACKS
def build_vectorstore() -> FAISS:
    """Loads documents, splits them, and creates a FAISS vectorstore using Gemini Embeddings."""

    # Determine paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Tries to find 'data' folder one level up
    data_folder = os.path.join(os.path.dirname(base_dir), "data")

    print(f"--- SYSTEM: Looking for documents in: {data_folder} ---")

    if not os.path.isdir(data_folder):
        # Fallback: Try current directory if the upper directory fails
        data_folder = os.path.join(base_dir, "data")
        if not os.path.isdir(data_folder):
            raise RuntimeError(
                f"Data folder not found at {data_folder}. Please ensure your 'data' folder exists."
            )

    # Load Documents
    docs: List[Document] = []
    for root, dirs, files in os.walk(data_folder):
        for fname in files:
            if fname.endswith(".txt"):
                file_path = os.path.join(root, fname)
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    loaded_docs = loader.load()
                    # Add metadata so we know which file a chunk came from
                    for doc in loaded_docs:
                        doc.metadata["source"] = fname
                    docs.extend(loaded_docs)
                    print(f"--- SYSTEM: Loaded file: {fname}")
                except Exception as e:
                    print(f"Error loading {fname}: {e}")

    if not docs:
        # Fallback for empty data folder to prevent crash (optional safety)
        print("--- WARNING: No .txt documents found. Initializing empty store. ---")
        return FAISS.from_texts(
            ["Empty store"],
            GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", google_api_key=GEMINI_API_KEY
            ),
        )

    # Split Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Initialize Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=GEMINI_API_KEY
    )

    # Create Vector Store
    print("--- SYSTEM: Generating embeddings with Gemini... this may take a moment.")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# Initialize Vector Store
try:
    VECTORSTORE = build_vectorstore()
except Exception as e:
    print(f"CRITICAL ERROR: Could not build vectorstore. {e}")
    VECTORSTORE = None


# STATE DEFINITION
class RAGState(TypedDict):
    """The state schema for the LangGraph workflow."""

    query: str
    history: list  # short-term conversation memory
    context: str
    answer: str
    source_documents: List[Document]
    web_sources: list
    route_decision: str  # For router logic


# HELPER FUNCTIONS google sources extraction & retrieval with threshold

import requests
from urllib.parse import urlparse


def resolve_redirect_url(redirect_url: str, timeout=5) -> str:
    """
    Follows a redirect URL to get the final destination.
    Returns the real URL or the original if resolution fails.
    """
    if not redirect_url.startswith("https://vertexaisearch.cloud.google.com"):
        return redirect_url  # Already a real URL

    try:
        response = requests.head(redirect_url, allow_redirects=True, timeout=timeout)
        resolved = response.url
        print(f"   Resolved: {redirect_url[:60]}... -> {resolved}")
        return resolved
    except requests.RequestException as e:
        print(f"   Failed to resolve redirect: {e}")
        return redirect_url  # Fallback to original


def extract_google_sources(response):
    """Safely extracts and resolves web sources from Gemini API response."""
    sources = []
    try:
        for cand in getattr(response, "candidates", []) or []:
            gm = getattr(cand, "grounding_metadata", None)
            if not gm:
                continue
            for chunk in getattr(gm, "grounding_chunks", []) or []:
                web = getattr(chunk, "web", None)
                if not web:
                    continue
                url = getattr(web, "uri", None)
                if not url:
                    continue

                real_url = resolve_redirect_url(url)

                title = getattr(web, "title", "") or real_url
                sources.append({"url": real_url, "title": title})
    except Exception as e:
        print(f"Error extracting Google sources: {e}")

    unique = {s["url"]: s["title"] for s in sources}
    return [{"url": u, "title": t} for u, t in unique.items()]


def retrieve_with_threshold(query, k=4, min_score=0.5):
    """Retrieves relevant docs, filters by score, and PRINTS DEBUG INFO."""
    if not VECTORSTORE:
        return []

    print(f"\n\n=== RETRIEVAL DEBUG: Query='{query}' ===")
    try:
        # FAISS returns L2 distance (lower is better).
        raw = VECTORSTORE.similarity_search_with_score(query, k=k)

        passing_docs = []

        for doc, score in raw:
            # Convert L2 Distance to an approximate cosine similarity
            similarity = 1 - (score**2 / 2)
            filename = doc.metadata.get("source", "unknown")

            print(
                f"   [File: {filename}] Raw Distance: {score:.4f} | Converted Similarity: {similarity:.4f}"
            )

            if similarity >= min_score:
                passing_docs.append(doc)
            else:
                print(f"   -> REJECTED (Threshold {min_score})")

        return passing_docs

    except Exception as e:
        print(f"Retrieval failed: {e}")
        return []


def format_history(history):
    """Format chat history (list of {role, text}) into a readable transcript."""
    lines = []
    for m in history or []:
        role = m.get("role", "user")
        text = m.get("text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


# AGENTIC NODES


def router_node(state: RAGState) -> RAGState:
    """
    Node 1 (ROUTER): Decides if we need to retrieve docs or just chat.
    """
    print(f"\n--- NODE: Router (Analyzing Query) ---")
    query = state["query"]

    # Simple check or LLM call to decide routing
    # Using a lightweight LLM call here ensures it's 'Agentic'
    prompt = f"""
    Analyze the user query: "{query}".
    If the user is asking for specific information, facts, or details that might be in a document, output "retrieve".
    If the user is just saying "hello", "thanks", "goodbye", or engaging in small talk, output "chat".
    Output ONLY the word "retrieve" or "chat".
    """
    try:
        # Using 2.0-flash for speed/cost
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        decision = resp.text.strip().lower()
    except:
        decision = "retrieve"  # Default fallback

    if "chat" in decision:
        print(">>> ROUTING DECISION: Simple chat detected. Skipping retrieval.")
        return {**state, "route_decision": "chat"}
    else:
        print(">>> ROUTING DECISION: Information query detected. Routing to Retriever.")
        return {**state, "route_decision": "retrieve"}


def retrieve_node(state: RAGState) -> RAGState:
    """Node 2: Retrieves documents from the vector store."""
    print(f"--- NODE: Retrieve ---")

    # Adjust min_score here to tweak sensitivity (Using your original logic)
    docs = retrieve_with_threshold(state["query"], k=4, min_score=0.4)
    context = "\n\n".join(doc.page_content for doc in docs)

    if not docs:
        print("\n>>> RETRIEVER: No relevant local chunks found above threshold.\n")
    else:
        print(f"\n>>> RETRIEVER: Found {len(docs)} relevant local chunks.\n")

    return {
        **state,
        "context": context,
        "source_documents": docs,
    }


def generate_node(state: RAGState) -> RAGState:
    """Node 3: Generates answer using Context (if any) + History + Google Search."""
    print(f"--- NODE: Generate ---")

    conversation = format_history(state.get("history", []))

    # Configuration to enable Google Search grounding for the LLM
    grounding_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
    generate_config = genai_types.GenerateContentConfig(tools=[grounding_tool])

    # Optimized Agentic Prompt
    prompt = f"""
    You are a helpful and knowledgeable assistant.

    # ROLE
    Your goal is to provide accurate answers by synthesizing information from the provided conversation history, retrieved local context, and external Google Search results if necessary.

    # CONVERSATION HISTORY
    {conversation}

    # RETRIEVED CONTEXT (Local Documents)
    {state.get('context', '')}

    # USER QUESTION
    {state['query']}

    # INSTRUCTIONS
    1. **CONTEXTUALIZE**: Use the conversation history to maintain context and resolve references (e.g., "it", "that").
    2. **PRIORITY**: ALWAYS start by using the information in the "RETRIEVED CONTEXT" section above to answer the user's question.
    3. **FALLBACK**: If the local context is empty, irrelevant, or does not fully answer the question, YOU MUST use the Google Search tool to find the missing information.
    4. **COMBINE**: If the local context only answers part of the question, use Google Search to fill in the details, then combine both sources into a single cohesive response.
    5. **STYLE**: Be clear, concise, and professional.

    Answer:
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Or gemini-2.5-flash if available
            contents=prompt,
            config=generate_config,
        )
        answer_text = response.text if hasattr(response, "text") else str(response)
        web_sources = extract_google_sources(response)

        if web_sources:
            print(
                f">>> MODEL ACTION: Used Google Search Grounding. Found {len(web_sources)} external links."
            )

        return {
            **state,
            "answer": answer_text,
            "web_sources": web_sources,
        }
    except Exception as e:
        print(f"Generation error: {e}")
        return {
            **state,
            "answer": "I encountered an error generating the response.",
            "web_sources": [],
        }


def memory_node(state: RAGState) -> RAGState:
    """
    Node 4 (MEMORY): explicitly updates/saves the history state.
    """
    print("--- NODE: Memory (Updating State) ---")

    # Take existing history (from frontend) or empty list
    history = state.get("history", [])

    # Append the latest assistant answer as a new turn
    if state.get("answer"):
        history = history + [{"role": "assistant", "text": state["answer"]}]

    # Return updated state (so the graph now contains the new history)
    return {
        **state,
        "history": history,
    }


# BUILD THE AGENTIC GRAPH


def get_rag_graph() -> StateGraph:
    """Defines and returns the LangGraph workflow."""
    graph = StateGraph(RAGState)

    # Add Nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("memory", memory_node)

    # Set Entry Point
    graph.set_entry_point("router")

    # Conditional Edges from Router
    def route_logic(state: RAGState):
        if state.get("route_decision") == "chat":
            return "generate"  # Skip retrieval
        return "retrieve"  # Do retrieval

    graph.add_conditional_edges(
        "router", route_logic, {"retrieve": "retrieve", "generate": "generate"}
    )

    # Normal Edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "memory")
    graph.add_edge("memory", END)

    return graph
