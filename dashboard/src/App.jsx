{ /* Maaz Bobat, Saaram Rashidi, MD Sazid, Sun Hung Tsang, Yehor Valesiuk*/ }
import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import Navbar from "./components/Navbar"
import PotholeDetector from "./pages/PotholeDetector"
import RagChat from "./components/RagChat"
import "bootstrap/dist/css/bootstrap.min.css"
import "./App.css"

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <Routes>
          <Route path="/" element={<PotholeDetector />} />
          <Route path="/ragchat" element={<RagChat />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
