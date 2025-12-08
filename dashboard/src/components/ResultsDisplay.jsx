import { Card, Spinner, Alert, Badge } from "react-bootstrap"

export default function ResultsDisplay({ results, loading, error }) {
  if (loading) {
    return (
      <Card className="fade-in">
        <Card.Body className="text-center py-5">
          <Spinner animation="border" role="status" style={{ width: "3rem", height: "3rem" }}>
            <span className="visually-hidden">Loading results...</span>
          </Spinner>
          <p className="mt-3 text-muted fs-5">Analyzing image for potholes...</p>
          <p className="text-muted small">This may take a few moments</p>
        </Card.Body>
      </Card>
    )
  }

  if (error) {
    return (
      <Alert variant="danger" className="fade-in" role="alert">
        <Alert.Heading>❌ Error</Alert.Heading>
        <p>{error}</p>
      </Alert>
    )
  }

  if (!results) {
    return (
      <Alert variant="info" className="fade-in" role="status">
        <div className="text-center py-3">
          <p className="fs-5 mb-2">👆 Upload an image to detect potholes</p>
          <p className="text-muted small mb-0">Select an image of a road surface and click detect</p>
        </div>
      </Alert>
    )
  }

  return (
    <Card className="fade-in">
      <Card.Header as="h5" className="d-flex justify-content-between align-items-center">
        <span>{results.hasPothole ? "🔴 Pothole Detected!" : "✅ No Potholes Detected"}</span>
        <Badge bg="light" text="dark" className="fs-6 me-2">
          {(results.confidence * 100).toFixed(1)}% Confidence
        </Badge>
        <Badge bg="danger" className="fs-6">
          {results.count} Potholes
        </Badge>
      </Card.Header>
      <Card.Body>
        {results.hasPothole ? (
          <>
            <Alert variant="warning" className="mb-3">
              <strong>⚠️ Attention:</strong> One or more potholes have been detected in this image.
            </Alert>
            {(results.annotatedImage || results.originalImage) && (
              <div className="text-center">
                <img
                  src={results.annotatedImage ? results.annotatedImage : results.originalImage}
                  alt="Analyzed road surface"
                  style={{
                    width: "100%",
                    maxHeight: "600px",
                    objectFit: "contain",
                    borderRadius: "8px"
                  }}
                  className="shadow-sm"
                />
              </div>
            )}
          </>
        ) : (
          <>
            <Alert variant="success" className="mb-3">
              <Alert.Heading>✅ All Clear!</Alert.Heading>
              <p className="mb-0">No potholes were detected in this image. The road surface appears to be in good condition.</p>
            </Alert>
            {results.originalImage && (
              <div className="text-center">
                <img
                  src={results.originalImage}
                  alt="Analyzed road surface"
                  style={{
                    width: "100%",
                    maxHeight: "600px",
                    objectFit: "contain",
                    borderRadius: "8px"
                  }}
                  className="shadow-sm"
                />
              </div>
            )}
          </>
        )}
      </Card.Body>
    </Card>
  )
}
