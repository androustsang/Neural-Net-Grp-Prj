import React, { useState } from 'react';
import { Card, Form, Button, Alert, Spinner, Row, Col } from 'react-bootstrap';

export default function PredictionForm({ onSubmit, loading }) {
  const [formData, setFormData] = useState({
    textInput: '',
    imageFile: null,
    tabularData: ''
  });
  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    if (name === 'imageFile') {
      setFormData(prev => ({ ...prev, [name]: files[0] }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
    setErrors(prev => ({ ...prev, [name]: '' }));
  };

  const validate = () => {
    const newErrors = {};
    if (!formData.textInput.trim() && !formData.imageFile && !formData.tabularData.trim()) {
      newErrors.general = 'Please provide at least one type of input';
    }
    return newErrors;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const validationErrors = validate();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
    onSubmit(formData);
  };

  const handleClear = () => {
    setFormData({
      textInput: '',
      imageFile: null,
      tabularData: ''
    });
    setErrors({});
  };

  return (
    <Card className="mb-4 fade-in">
      <Card.Header as="h5">📝 Input Data</Card.Header>
      <Card.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-4" controlId="textInput">
            <Form.Label className="fw-semibold">Text Input</Form.Label>
            <Form.Control
              as="textarea"
              rows={5}
              name="textInput"
              value={formData.textInput}
              onChange={handleChange}
              placeholder="Enter text for sentiment analysis, classification, or other NLP tasks..."
              disabled={loading}
              aria-label="Text input for analysis"
              className="shadow-sm"
            />
            <Form.Text className="text-muted">
              Enter any text you'd like to analyze (e.g., product reviews, social media posts, articles)
            </Form.Text>
          </Form.Group>

          <Form.Group className="mb-4" controlId="imageFile">
            <Form.Label className="fw-semibold">Image Upload</Form.Label>
            <Form.Control
              type="file"
              name="imageFile"
              onChange={handleChange}
              accept="image/*"
              disabled={loading}
              aria-label="Upload image file"
              className="shadow-sm"
            />
            {formData.imageFile && (
              <div className="mt-2 p-2 bg-light rounded">
                <small className="text-success fw-semibold">
                  ✓ Selected: {formData.imageFile.name}
                </small>
              </div>
            )}
            <Form.Text className="text-muted">
              Upload images for classification, object detection, or visual analysis
            </Form.Text>
          </Form.Group>

          <Form.Group className="mb-4" controlId="tabularData">
            <Form.Label className="fw-semibold">Tabular Data (CSV format)</Form.Label>
            <Form.Control
              as="textarea"
              rows={4}
              name="tabularData"
              value={formData.tabularData}
              onChange={handleChange}
              placeholder="feature1,feature2,feature3&#10;1.2,3.4,5.6&#10;2.3,4.5,6.7"
              disabled={loading}
              aria-label="Tabular data in CSV format"
              className="shadow-sm font-monospace"
            />
            <Form.Text className="text-muted">
              Enter structured data in CSV format (comma-separated values)
            </Form.Text>
          </Form.Group>

          {errors.general && (
            <Alert variant="danger" className="fade-in" role="alert">
              <strong>⚠️ Error:</strong> {errors.general}
            </Alert>
          )}

          <Row className="mt-4">
            <Col>
              <Button 
                variant="primary" 
                type="submit" 
                disabled={loading}
                className="w-100 shadow-sm"
                aria-label="Submit data for prediction"
              >
                {loading ? (
                  <>
                    <Spinner
                      as="span"
                      animation="border"
                      size="sm"
                      role="status"
                      aria-hidden="true"
                      className="me-2"
                    />
                    Processing...
                  </>
                ) : (
                  <>🚀 Submit for Prediction</>
                )}
              </Button>
            </Col>
            <Col xs="auto">
              <Button 
                variant="outline-secondary" 
                onClick={handleClear}
                disabled={loading}
                className="shadow-sm"
              >
                Clear
              </Button>
            </Col>
          </Row>
        </Form>
      </Card.Body>
    </Card>
  );
}
