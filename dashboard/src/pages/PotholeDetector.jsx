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

        // Simulate processing delay
        setTimeout(() => {
            // Randomly determine if pothole is detected (70% chance of detection)
            const hasPothole = Math.random() > 0.3

            // Random confidence between 75% and 95%
            const confidence = 0.75 + Math.random() * 0.2

            // Convert uploaded image to base64 for display
            const reader = new FileReader()
            reader.onloadend = () => {
                const base64Image = reader.result.split(',')[1]

                setResults({
                    hasPothole: hasPothole,
                    annotatedImage: hasPothole ? base64Image : null,
                    confidence: confidence,
                    originalImage: reader.result,
                })
                setLoading(false)
            }
            reader.readAsDataURL(imageFile)
        }, 2000)
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
