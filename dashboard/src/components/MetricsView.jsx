import React from 'react';
import { Card, Row, Col, Alert, Table, Spinner, Badge } from 'react-bootstrap';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function MetricsView({ metrics, loading, error }) {
  if (loading) {
    return (
      <Card className="fade-in">
        <Card.Body className="text-center py-5">
          <Spinner animation="border" role="status" style={{ width: '3rem', height: '3rem' }}>
            <span className="visually-hidden">Loading metrics...</span>
          </Spinner>
          <p className="mt-3 text-muted fs-5">Loading model metrics...</p>
        </Card.Body>
      </Card>
    );
  }

  if (error) {
    return <Alert variant="danger" className="fade-in" role="alert">❌ {error}</Alert>;
  }

  if (!metrics) {
    return <Alert variant="info" className="fade-in" role="status">No metrics available</Alert>;
  }

  return (
    <>
      <Card className="mb-4 fade-in">
        <Card.Header as="h5">🎯 Model Performance Metrics</Card.Header>
        <Card.Body>
          <Row className="g-4">
            <Col md={3}>
              <div className="text-center p-4 bg-light rounded-3 shadow-sm">
                <h6 className="text-muted mb-2">Accuracy</h6>
                <h3 className="fw-bold text-primary mb-2">
                  {(metrics.accuracy * 100).toFixed(1)}%
                </h3>
                <Badge bg="primary">Excellent</Badge>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center p-4 bg-light rounded-3 shadow-sm">
                <h6 className="text-muted mb-2">Precision</h6>
                <h3 className="fw-bold text-success mb-2">
                  {(metrics.precision * 100).toFixed(1)}%
                </h3>
                <Badge bg="success">High</Badge>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center p-4 bg-light rounded-3 shadow-sm">
                <h6 className="text-muted mb-2">Recall</h6>
                <h3 className="fw-bold text-info mb-2">
                  {(metrics.recall * 100).toFixed(1)}%
                </h3>
                <Badge bg="info">Good</Badge>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center p-4 bg-light rounded-3 shadow-sm">
                <h6 className="text-muted mb-2">F1 Score</h6>
                <h3 className="fw-bold text-warning mb-2">
                  {(metrics.f1Score * 100).toFixed(1)}%
                </h3>
                <Badge bg="warning" text="dark">Strong</Badge>
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      <Row>
        <Col lg={6}>
          <Card className="mb-4 fade-in">
            <Card.Header as="h6">📈 Training History - Accuracy</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={metrics.trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis 
                    dataKey="epoch" 
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }}
                    domain={[0, 1]}
                  />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="rgb(132, 132, 140)" 
                    strokeWidth={3}
                    dot={{ fill: 'rgb(84, 84, 92)', r: 5 }}
                    activeDot={{ r: 7 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>

        <Col lg={6}>
          <Card className="mb-4 fade-in">
            <Card.Header as="h6">📉 Training History - Loss</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={metrics.trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis 
                    dataKey="epoch" 
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#dc3545" 
                    strokeWidth={3}
                    dot={{ fill: '#c82333', r: 5 }}
                    activeDot={{ r: 7 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Card className="fade-in">
        <Card.Header as="h6">🎲 Confusion Matrix</Card.Header>
        <Card.Body>
          <Table bordered responsive className="text-center shadow-sm">
            <thead className="table-light">
              <tr>
                <th></th>
                {metrics.classes.map(cls => (
                  <th key={cls}><strong>Predicted {cls}</strong></th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.confusionMatrix.map((row, i) => (
                <tr key={i}>
                  <th className="table-light"><strong>Actual {metrics.classes[i]}</strong></th>
                  {row.map((val, j) => (
                    <td 
                      key={j}
                      style={{ 
                        backgroundColor: i === j ? '#d4edda' : '#f8d7da',
                        fontWeight: i === j ? 'bold' : 'normal',
                        fontSize: '1.1rem'
                      }}
                    >
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
        </Card.Body>
      </Card>
    </>
  );
}
