import { useState } from "react"
import PredictionForm from "../components/PredictionForm"
import ResultsDisplay from "../components/ResultsDisplay"
import { Container, Row, Col } from "react-bootstrap"

export default function PotholeDetector() {
    const [results, setResults] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handlePredictionSubmit = async (imageFile) => {
        setLoading(true)
        setError(null)

        try {
            const formData = new FormData()
            formData.append("image", imageFile)

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                throw new Error(errorData.error || `Server error: ${response.status}`)
            }

            const data = await response.json()

            // Read the original image for display
            const reader = new FileReader()
            reader.onloadend = () => {
                setResults({
                    hasPothole: data.prediction === "pothole",
                    annotatedImage: null, // Backend doesn't return annotated image yet
                    confidence: data.confidence,
                    originalImage: reader.result,
                })
                setLoading(false)
            }
            reader.readAsDataURL(imageFile)

        } catch (err) {
            console.error("Prediction error:", err)
            setError(err.message || "Failed to get prediction")
            setLoading(false)
        }
    }

    return (
        <Container fluid className="py-4 h-100">
            <Row className="h-100">
                <Col lg={6} className="mb-3 mb-lg-0">
                    <PredictionForm onSubmit={handlePredictionSubmit} loading={loading} />
                </Col>
                <Col lg={6}>
                    <ResultsDisplay results={results} loading={loading} error={error} />
                </Col>
            </Row>
        </Container>
    )
}
