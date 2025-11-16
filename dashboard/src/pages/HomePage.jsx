import React, { useState } from 'react';
import { Container, Row, Col, Tab, Nav, Alert } from 'react-bootstrap';
import PredictionForm from '../components/PredictionForm';
import Chatbot from '../components/Chatbot';
import { api } from '../services/api';

export default function HomePage({ onPredictionComplete }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFormSubmit = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const results = await api.predict(formData);
      const summary = await api.generateSummary(formData);
      onPredictionComplete({ results, summary, formData });
    } catch (err) {
      setError(err.message || 'Prediction failed. Please try again.');
      console.error('Prediction failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="py-5">
      <div className="text-center mb-5">
        <h1 className="display-4 fw-bold gradient-text mb-3">Welcome to ML Dashboard</h1>
        <p className="lead text-muted">
          Analyze your data with state-of-the-art machine learning models
        </p>
      </div>

      {error && (
        <Alert variant="danger" dismissible onClose={() => setError(null)} className="fade-in">
          <strong>Error:</strong> {error}
        </Alert>
      )}

      <Tab.Container defaultActiveKey="predict">
        <Row>
          <Col>
            <Nav variant="pills" className="mb-4 justify-content-center">
              <Nav.Item>
                <Nav.Link eventKey="predict" className="px-4 py-2 fw-semibold">
                  🎯 Make Prediction
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="chat" className="px-4 py-2 fw-semibold">
                  💬 AI Assistant
                </Nav.Link>
              </Nav.Item>
            </Nav>
          </Col>
        </Row>

        <Tab.Content>
          <Tab.Pane eventKey="predict">
            <PredictionForm onSubmit={handleFormSubmit} loading={loading} />
          </Tab.Pane>

          <Tab.Pane eventKey="chat">
            <Chatbot />
          </Tab.Pane>
        </Tab.Content>
      </Tab.Container>
    </Container>
  );
}
