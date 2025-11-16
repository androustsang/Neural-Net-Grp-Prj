import React, { useState } from 'react';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import ResultsPage from './pages/ResultsPage';

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [predictionData, setPredictionData] = useState(null);

  const handlePredictionComplete = (data) => {
    setPredictionData(data);
    setCurrentPage('results');
  };

  const handleNavigateHome = () => {
    setCurrentPage('home');
  };

  return (
    <div className="min-vh-100" style={{ backgroundColor: '#f8f9fa' }}>
      <Navbar />
      {currentPage === 'home' ? (
        <HomePage onPredictionComplete={handlePredictionComplete} />
      ) : (
        <ResultsPage 
          predictionData={predictionData} 
          onBackToHome={handleNavigateHome}
        />
      )}
    </div>
  );
}
