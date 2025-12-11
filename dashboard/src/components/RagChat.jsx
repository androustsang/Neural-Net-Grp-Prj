{/* Maaz Bobat, Saaram Rashidi, MD Sazid, Sun Hung Tsang, Yehor Valesiuk*/ }
import { useState, useRef, useEffect } from "react"
import { Container } from "react-bootstrap"
import "../styles/RagChat.css"

export default function RagChat() {
    const [question, setQuestion] = useState("")
    const [messages, setMessages] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState("")

    const chatEndRef = useRef(null)

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])

    const askQuestion = async (e) => {
        e.preventDefault()
        const trimmed = question.trim()
        if (!trimmed) return

        const newUserMessage = { role: "user", text: trimmed }
        const historyToSend = [...messages, newUserMessage]

        setMessages(historyToSend)
        setQuestion("")
        setError("")
        setLoading(true)

        try {
            const response = await fetch("/rag/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: trimmed,
                    history: historyToSend,
                }),
            })

            const data = await response.json()

            if (response.ok) {
                const answer = data["Model response"] || ""
                const context = data["Retrieved context"] || ""
                const sources = data.sources || []
                const fullUpdatedHistory = data.history || []

                if (answer) {
                    const finalMessages = fullUpdatedHistory.map((m, index) => {
                        if (index === fullUpdatedHistory.length - 1 && m.role === "assistant") {
                            return {
                                ...m,
                                sources: sources,
                                context: context,
                            }
                        }
                        return m
                    })

                    setMessages(finalMessages)
                }
            } else {
                setError(data.error || "Something went wrong.")
            }
        } catch (err) {
            console.error(err)
            setError("Could not connect to backend.")
        } finally {
            setLoading(false)
        }
    }

    const formatSource = (src) => {
        if (typeof src !== "string") return src

        try {
            const url = new URL(src)
            const hostname = url.hostname.replace("www.", "")
            return hostname
        } catch {
            return src
        }
    }

    const makeSourceLink = (src) => {
        if (typeof src !== "string") return null
        try {
            new URL(src)
            return src
        } catch {
            return null
        }
    }

    return (
        <Container fluid className="py-4 fade-in">
            <div className="ragchat-container">
                <header className="ragchat-header">
                    <h1>Agentic RAG Assistant</h1>
                </header>

                <div className="ragchat-messages">
                    {messages.map((m, idx) => (
                        <div key={idx} className={`ragchat-message ${m.role}`}>
                            <div className="ragchat-meta">{m.role === "user" ? "You" : "Assistant"}</div>
                            <div style={{ whiteSpace: "pre-wrap" }}>{m.text}</div>

                            {m.role === "assistant" && m.context && (
                                <details className="ragchat-details">
                                    <summary className="ragchat-summary">View Retrieved Context (Debug)</summary>
                                    <div className="ragchat-context-box">{m.context}</div>
                                </details>
                            )}

                            {m.sources && m.sources.length > 0 && (
                                <div className="ragchat-sources">
                                    <strong>Sources:</strong>
                                    <ul>
                                        {m.sources.map((src, srcIdx) => {
                                            const label = formatSource(src)
                                            const linkUrl = makeSourceLink(src)
                                            return (
                                                <li key={srcIdx}>
                                                    {linkUrl ? (
                                                        <a href={linkUrl} target="_blank" rel="noreferrer" title={src}>
                                                            {label}
                                                        </a>
                                                    ) : (
                                                        label
                                                    )}
                                                </li>
                                            )
                                        })}
                                    </ul>
                                </div>
                            )}
                        </div>
                    ))}

                    {error && <div className="ragchat-error">{error}</div>}
                    {loading && <div className="ragchat-loading">Assistant is thinking...</div>}
                    <div ref={chatEndRef} />
                </div>

                <form onSubmit={askQuestion} className="ragchat-form">
                    <input
                        type="text"
                        placeholder="Type your message..."
                        autoComplete="off"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        disabled={loading}
                        className="ragchat-input"
                    />
                    <button type="submit" disabled={loading} className="ragchat-button">
                        {loading ? "..." : "Send"}
                    </button>
                </form>
            </div>
        </Container>
    )
}
