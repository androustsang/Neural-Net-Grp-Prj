import { useState, useRef, useEffect } from "react"
import { Card, Form, Button, Alert, Spinner } from "react-bootstrap"

export default function PredictionForm({ onSubmit, loading }) {
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [error, setError] = useState("")
  const [box, setBox] = useState({ x: 0, y: 0, w: 100, h: 100 })
  const imgRef = useRef(null)
  const [imgLoaded, setImgLoaded] = useState(false)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      setError("")

      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
        setImgLoaded(false) // Reset loaded state for new image
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
    setImgLoaded(false)
    setBox({ x: 0, y: 0, w: 100, h: 100 })
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
            <div className="mb-3">
              <div className="position-relative d-inline-block" style={{ width: "100%", maxWidth: "640px" }}>
                <img
                  ref={imgRef}
                  src={imagePreview}
                  alt="Preview"
                  onLoad={() => setImgLoaded(true)}
                  style={{
                    width: "100%",
                    aspectRatio: "1 / 1",
                    objectFit: "fill",
                    borderRadius: "8px"
                  }}
                  className="shadow-sm"
                />
                {imgLoaded && imgRef.current && (
                  <div
                    style={{
                      position: "absolute",
                      border: "3px solid #00ff00",
                      backgroundColor: "rgba(0, 255, 0, 0.2)",
                      left: `${(box.x / 640) * imgRef.current.clientWidth}px`,
                      top: `${(box.y / 640) * imgRef.current.clientHeight}px`,
                      width: `${(box.w / 640) * imgRef.current.clientWidth}px`,
                      height: `${(box.h / 640) * imgRef.current.clientHeight}px`,
                      pointerEvents: "none"
                    }}
                  />
                )}
              </div>

              <div className="mt-3 p-3 bg-dark bg-opacity-10 rounded border border-secondary">
                <h6 className="mb-2">Manual Box Coordinates (Standard 640x640)</h6>
                <div className="row g-2">
                  <div className="col-3">
                    <Form.Group controlId="boxX">
                      <Form.Label className="small mb-1">X</Form.Label>
                      <Form.Control
                        type="number"
                        value={box.x}
                        onChange={(e) => setBox({ ...box, x: Number(e.target.value) })}
                        size="sm"
                      />
                    </Form.Group>
                  </div>
                  <div className="col-3">
                    <Form.Group controlId="boxY">
                      <Form.Label className="small mb-1">Y</Form.Label>
                      <Form.Control
                        type="number"
                        value={box.y}
                        onChange={(e) => setBox({ ...box, y: Number(e.target.value) })}
                        size="sm"
                      />
                    </Form.Group>
                  </div>
                  <div className="col-3">
                    <Form.Group controlId="boxW">
                      <Form.Label className="small mb-1">Width</Form.Label>
                      <Form.Control
                        type="number"
                        value={box.w}
                        onChange={(e) => setBox({ ...box, w: Number(e.target.value) })}
                        size="sm"
                      />
                    </Form.Group>
                  </div>
                  <div className="col-3">
                    <Form.Group controlId="boxH">
                      <Form.Label className="small mb-1">Height</Form.Label>
                      <Form.Control
                        type="number"
                        value={box.h}
                        onChange={(e) => setBox({ ...box, h: Number(e.target.value) })}
                        size="sm"
                      />
                    </Form.Group>
                  </div>
                </div>
              </div>
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
