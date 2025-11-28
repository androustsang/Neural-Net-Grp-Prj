import { useState, useRef, useEffect } from "react";

function RagChat() {
    const [question, setQuestion] = useState("");
    // State now stores the full, current conversation history
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    // Auto-scroll to bottom on new message
    const chatEndRef = useRef(null);
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const askQuestion = async (e) => {
        e.preventDefault();
        const trimmed = question.trim();
        if (!trimmed) return;

        // 1. Optimistically add User Message for immediate display
        const newUserMessage = { role: "user", text: trimmed };
        // We send the current state + the new user message as the history
        const historyToSend = [...messages, newUserMessage];

        setMessages(historyToSend); // Update UI immediately
        setQuestion("");
        setError("");
        setLoading(true);

        try {
            // Send Request to correct Endpoint
            const response = await fetch("http://127.0.0.1:5000/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: trimmed,
                    history: historyToSend // Send the full history
                }),
            });

            const data = await response.json();

            if (response.ok) {
                // 2. Extract Data using the ASSIGNMENT-SPECIFIED keys
                // Must use bracket notation for keys with spaces
                const answer = data["Model response"] || "";
                const context = data["Retrieved context"] || "";
                const sources = data.sources || [];

                // 3. Use the full history returned by the backend (includes user message + agent response)
                const fullUpdatedHistory = data.history || [];

                if (answer) {
                    // Find the latest assistant message (which should be the last message in fullUpdatedHistory)
                    // and inject the debug/metadata (context, sources) into it for display.
                    const finalMessages = fullUpdatedHistory.map((m, index) => {
                        if (index === fullUpdatedHistory.length - 1 && m.role === "assistant") {
                            return {
                                ...m,
                                sources: sources,
                                context: context
                            };
                        }
                        return m;
                    });

                    setMessages(finalMessages);
                }
            } else {
                setError(data.error || "Something went wrong.");
            }
        } catch (err) {
            console.error(err);
            setError("Could not connect to backend.");
        } finally {
            setLoading(false);
        }
    };

    // NEW: Helper: Prettify sources (strings from backend)
    const formatSource = (src) => {
        if (typeof src !== 'string') return src; // Skip non-strings (future-proof)

        try {
            const url = new URL(src);
            const hostname = url.hostname.replace('www.', ''); // Clean "www."
            return hostname;
        } catch {
            return src; // Local filename or invalid URL → show full
        }
    };

    // NEW: Helper: Check if valid URL for linking
    const makeSourceLink = (src) => {
        if (typeof src !== 'string') return null;
        try {
            new URL(src); // Valid URL?
            return src;
        } catch {
            return null; // Not a URL
        }
    };

    return (
        <div className="app" style={styles.app}>
            <header style={styles.header}>
                <h1 style={{ margin: 0, fontSize: "1.25rem" }}>
                    Agentic RAG Assistant
                </h1>
            </header>

            <div id="chat" className="chat" style={styles.chatContainer}>
                {messages.map((m, idx) => (
                    <div
                        key={idx}
                        className={`message ${m.role}`}
                        style={{
                            ...styles.messageBubble,
                            ...(m.role === "user" ? styles.userBubble : styles.assistantBubble),
                        }}
                    >
                        <div style={styles.meta}>
                            {m.role === "user" ? "You" : "Assistant"}
                        </div>
                        <div style={{ whiteSpace: "pre-wrap" }}>{m.text}</div>

                        {/* Display Context/Sources only on the assistant's latest message (where the metadata was injected) */}
                        {m.role === "assistant" && m.context && (
                            <details style={styles.details}>
                                <summary style={styles.summary}>View Retrieved Context (Debug)</summary>
                                <div style={styles.contextBox}>
                                    {m.context}
                                </div>
                            </details>
                        )}

                        {/* UPDATED: Sources Section with prettified links */}
                        {m.sources && m.sources.length > 0 && (
                            <div style={styles.sourcesContainer}>
                                <strong>Sources:</strong>
                                <ul style={{ paddingLeft: '20px', margin: '0.5rem 0 0' }}>
                                    {m.sources.map((src, srcIdx) => {
                                        const label = formatSource(src);
                                        const linkUrl = makeSourceLink(src);
                                        return (
                                            <li key={srcIdx} style={{ marginBottom: '0.25rem' }}>
                                                {linkUrl ? (
                                                    <a
                                                        href={linkUrl}
                                                        target="_blank"
                                                        rel="noreferrer"
                                                        style={{
                                                            color: '#4f46e5',
                                                            textDecoration: 'underline',
                                                            fontSize: 'inherit'
                                                        }}
                                                        title={src} // Full URL on hover
                                                    >
                                                        {label}
                                                    </a>
                                                ) : (
                                                    label
                                                )}
                                            </li>
                                        );
                                    })}
                                </ul>
                            </div>
                        )}
                    </div>
                ))}

                {error && <div style={styles.error}>{error}</div>}
                {loading && <div style={styles.loading}>Assistant is thinking...</div>}
                <div ref={chatEndRef} />
            </div>

            <form onSubmit={askQuestion} style={styles.form}>
                <input
                    type="text"
                    placeholder="Type your message..."
                    autoComplete="off"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={loading}
                    style={styles.input}
                />
                <button type="submit" disabled={loading} style={styles.button}>
                    {loading ? "..." : "Send"}
                </button>
            </form>
        </div>
    );
}

// Extracted styles for cleanliness (unchanged)
const styles = {
    app: {
        fontFamily: "system-ui, sans-serif",
        background: "#0f172a",
        color: "#e5e7eb",
        maxWidth: "800px",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        marginInline: "auto",
    },
    header: {
        padding: "1rem 1.5rem",
        borderBottom: "1px solid #1f2937",
    },
    chatContainer: {
        flex: 1,
        padding: "1rem 1.5rem",
        overflowY: "auto",
    },
    messageBubble: {
        marginBottom: "0.75rem",
        maxWidth: "85%",
        padding: "0.8rem",
        borderRadius: "0.5rem",
    },
    userBubble: {
        marginLeft: "auto",
        background: "#4f46e5",
        color: "#ffffff",
    },
    assistantBubble: {
        marginRight: "auto",
        background: "#1e293b",
        color: "#e5e7eb",
        border: "1px solid #334155",
    },
    meta: {
        fontSize: "0.75rem",
        opacity: 0.7,
        marginBottom: "0.25rem",
        fontWeight: "bold"
    },
    details: {
        marginTop: "1rem",
        borderTop: "1px dashed #475569",
        paddingTop: "0.5rem",
    },
    summary: {
        cursor: "pointer",
        fontSize: "0.8rem",
        color: "#94a3b8",
        userSelect: "none",
    },
    contextBox: {
        marginTop: "0.5rem",
        background: "#0f172a",
        padding: "0.5rem",
        borderRadius: "0.25rem",
        fontSize: "0.75rem",
        fontFamily: "monospace",
        whiteSpace: "pre-wrap",
        maxHeight: "200px",
        overflowY: "auto",
        border: "1px solid #334155"
    },
    sourcesContainer: {
        marginTop: "1rem",
        fontSize: "0.85rem",
        borderTop: "1px dashed #475569",
        paddingTop: "0.5rem",
        color: "#cbd5e1"
    },
    error: {
        marginTop: "0.75rem",
        padding: "0.6rem 0.8rem",
        borderRadius: "0.5rem",
        background: "#7f1d1d",
        color: "#fecaca",
    },
    loading: {
        padding: "0.6rem 0.8rem",
        borderRadius: "0.5rem",
        background: "#2563eb",
        color: "#eff6ff",
        width: "fit-content",
        fontSize: "0.9rem",
        margin: "0.5rem 0 0 0",
    },
    form: {
        display: "flex",
        gap: "0.5rem",
        padding: "1rem",
        borderTop: "1px solid #1f2937",
        background: "#020617",
    },
    input: {
        flex: 1,
        padding: "0.7rem 1rem",
        borderRadius: "999px",
        border: "1px solid #334155",
        background: "#1e293b",
        color: "#e5e7eb",
    },
    button: {
        padding: "0.7rem 1.5rem",
        borderRadius: "999px",
        border: "none",
        background: "#22c55e",
        color: "#020617",
        fontWeight: 600,
        cursor: "pointer",
    }
};

export default RagChat;
