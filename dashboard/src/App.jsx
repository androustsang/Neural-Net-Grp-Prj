import React, { useState } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import ResultsPage from './pages/ResultsPage';
import RagChat from './components/RagChat';

export default function App() {

  const [predictionData, setPredictionData] = useState(null);

  const navigate = useNavigate();

  const handlePredictionComplete = (data) => {

    setPredictionData(data);

    navigate('/results');

  };

  const handleNavigateHome = () => {

    navigate('/');

  };

  return (
    <div className="min-vh-100" style={{ backgroundColor: '#f8f9fa' }}>
      <Navbar />
      <Routes>
        <Route
          path="/"
          element={
            <HomePage onPredictionComplete={handlePredictionComplete} />
          }
        />
        <Route
          path="/results"
          element={
            <ResultsPage
              predictionData={predictionData}
              onBackToHome={handleNavigateHome}
            />
          }
        />
        <Route path="/chat" element={<RagChat />} />
      </Routes>
    </div>
  );


}
