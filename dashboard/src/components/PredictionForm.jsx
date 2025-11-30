import { useState } from "react"
import { Card, Form, Button, Alert, Spinner } from "react-bootstrap"

export default function PredictionForm({ onSubmit, loading }) {
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [error, setError] = useState("")

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      setError("")

      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!imageFile) {
      setError("Please select an image to analyze")
      return
    }
    onSubmit(imageFile)
  }

  const handleClear = () => {
    setImageFile(null)
    setImagePreview(null)
    setError("")
    document.getElementById("imageFile").value = ""
  }

  return (
    <Card className="fade-in">
      <Card.Header as="h5">📷 Upload Image for Pothole Detection</Card.Header>
      <Card.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3" controlId="imageFile">
            <Form.Label>Select Image</Form.Label>
            <Form.Control
              type="file"
              name="imageFile"
              onChange={handleImageChange}
              accept="image/*"
              disabled={loading}
              aria-label="Upload image file"
            />
            <Form.Text className="text-muted">
              Upload an image of road surface to detect potholes
            </Form.Text>
          </Form.Group>

          {imagePreview && (
            <div className="mb-3 text-center">
              <img
                src={imagePreview}
                alt="Preview"
                style={{
                  width: "100%",
                  maxHeight: "400px",
                  objectFit: "contain",
                  borderRadius: "8px"
                }}
                className="shadow-sm"
              />
            </div>
          )}

          {error && (
            <Alert variant="danger" className="fade-in mb-3" role="alert">
              <strong>⚠️ Error:</strong> {error}
            </Alert>
          )}

          <div className="d-flex gap-2">
            <Button
              variant="primary"
              type="submit"
              disabled={loading || !imageFile}
              className="flex-grow-1"
              aria-label="Submit image for detection"
            >
              {loading ? (
                <>
                  <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" className="me-2" />
                  Detecting Potholes...
                </>
              ) : (
                <>🚀 Detect Potholes</>
              )}
            </Button>
            <Button
              variant="outline-secondary"
              onClick={handleClear}
              disabled={loading}
            >
              Clear
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  )
}
