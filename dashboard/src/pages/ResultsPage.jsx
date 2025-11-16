import React, { useEffect, useState } from 'react';
import { Container, Row, Col, Tab, Nav, Card, Alert, Spinner, Button } from 'react-bootstrap';
import ResultsDisplay from '../components/ResultsDisplay';
import MetricsView from '../components/MetricsView';
import Chatbot from '../components/Chatbot';
import { api } from '../services/api';

export default function ResultsPage({ predictionData, onBackToHome }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getMetrics();
      setMetrics(data);
    } catch (err) {
      setError(err.message || 'Failed to load metrics');
    } finally {
      setLoading(false);
    }
  };

  if (!predictionData) {
    return (
      <Container className="py-5">
        <Alert variant="info" className="text-center fade-in">
          <Alert.Heading>No Prediction Data</Alert.Heading>
          <p>Please make a prediction first to see results.</p>
          <Button variant="primary" onClick={onBackToHome}>Go to Home</Button>
        </Alert>
      </Container>
    );
  }

  return (
    <Container className="py-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2 className="gradient-text fw-bold mb-0">Analysis Results</h2>
        <Button variant="outline-secondary" onClick={onBackToHome} className="shadow-sm">
          ← Back to Home
        </Button>
      </div>

      <Tab.Container defaultActiveKey="results">
        <Row>
          <Col>
            <Nav variant="tabs" className="mb-4">
              <Nav.Item>
                <Nav.Link eventKey="results" className="fw-semibold">
                  📊 Prediction Results
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="insights" className="fw-semibold">
                  💡 AI Insights
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="metrics" className="fw-semibold">
                  📈 Model Metrics
                </Nav.Link>
              </Nav.Item>
            </Nav>
          </Col>
        </Row>

        <Tab.Content>
          <Tab.Pane eventKey="results">
            <ResultsDisplay 
              results={predictionData.results}
              loading={false}
              error={null}
            />
          </Tab.Pane>

          <Tab.Pane eventKey="insights">
            <Card className="mb-4 fade-in">
              <Card.Header as="h5">💡 AI-Generated Insights</Card.Header>
              <Card.Body>
                {predictionData.summary ? (
                  <>
                    <div className="mb-4">
                      <h6 className="fw-semibold mb-3">Summary</h6>
                      <p className="lead">{predictionData.summary.summary}</p>
                    </div>
                    
                    <div>
                      <h6 className="fw-semibold mb-3">Key Insights</h6>
                      <ul className="list-unstyled">
                        {predictionData.summary.insights.map((insight, idx) => (
                          <li key={idx} className="mb-2 p-3 bg-light rounded shadow-sm">
                            <strong className="text-primary">✓</strong> {insight}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-4">
                    <Spinner animation="border" />
                    <p className="mt-3 text-muted">Generating insights...</p>
                  </div>
                )}
              </Card.Body>
            </Card>
            <Chatbot />
          </Tab.Pane>

          <Tab.Pane eventKey="metrics">
            <MetricsView 
              metrics={metrics}
              loading={loading}
              error={error}
            />
          </Tab.Pane>
        </Tab.Content>
      </Tab.Container>
    </Container>
  );
}
