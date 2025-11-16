import React from 'react';
import { Card, Row, Col, Spinner, Alert, Table, ProgressBar, Badge } from 'react-bootstrap';

export default function ResultsDisplay({ results, loading, error }) {
  if (loading) {
    return (
      <Card className="fade-in">
        <Card.Body className="text-center py-5">
          <Spinner animation="border" role="status" style={{ width: '3rem', height: '3rem' }}>
            <span className="visually-hidden">Loading results...</span>
          </Spinner>
          <p className="mt-3 text-muted fs-5">Analyzing your input...</p>
          <p className="text-muted small">This may take a few moments</p>
        </Card.Body>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="danger" className="fade-in" role="alert">
        <Alert.Heading>❌ Error</Alert.Heading>
        <p>{error}</p>
      </Alert>
    );
  }

  if (!results) {
    return (
      <Alert variant="info" className="fade-in" role="status">
        <div className="text-center py-3">
          <p className="fs-5 mb-2">👈 Submit data to see prediction results</p>
          <p className="text-muted small mb-0">Fill out the form and click submit to get started</p>
        </div>
      </Alert>
    );
  }

  return (
    <Card className="fade-in">
      <Card.Header as="h5">📊 Prediction Results</Card.Header>
      <Card.Body>
        <Row className="mb-4">
          <Col md={6}>
            <div className="text-center p-4 bg-light rounded-3 shadow-sm">
              <h6 className="text-muted mb-3">Prediction</h6>
              <h2 className="gradient-text fw-bold mb-0">{results.prediction}</h2>
              <Badge bg="primary" className="mt-2">Primary Prediction</Badge>
            </div>
          </Col>
          <Col md={6}>
            <div className="text-center p-4 bg-light rounded-3 shadow-sm">
              <h6 className="text-muted mb-3">Confidence Level</h6>
              <h2 className="text-success fw-bold mb-3">
                {(results.confidence * 100).toFixed(1)}%
              </h2>
              <ProgressBar 
                now={results.confidence * 100} 
                variant="success"
                aria-label={`Confidence level: ${(results.confidence * 100).toFixed(1)}%`}
                className="shadow-sm"
                style={{ height: '20px' }}
              />
            </div>
          </Col>
        </Row>

        <hr className="my-4" />

        <h6 className="mb-3 fw-semibold">📈 Class Probabilities</h6>
        <Table striped hover responsive className="shadow-sm">
          <thead>
            <tr>
              <th>Class</th>
              <th>Probability</th>
              <th width="50%">Distribution</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(results.probabilities).map(([cls, prob]) => (
              <tr key={cls}>
                <td className="fw-semibold">{cls}</td>
                <td>
                  <Badge bg={prob > 0.5 ? 'success' : 'secondary'}>
                    {(prob * 100).toFixed(2)}%
                  </Badge>
                </td>
                <td>
                  <ProgressBar 
                    now={prob * 100} 
                    variant={prob > 0.5 ? 'success' : 'secondary'}
                    aria-label={`${cls} probability: ${(prob * 100).toFixed(2)}%`}
                    style={{ height: '25px' }}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card.Body>
    </Card>
  );
}
