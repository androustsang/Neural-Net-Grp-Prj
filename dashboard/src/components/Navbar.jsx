import { Navbar as BSNavbar, Container, Nav } from "react-bootstrap"
import { Link, useLocation } from "react-router-dom"

export default function Navbar() {
  const location = useLocation()

  return (
    <BSNavbar variant="dark" expand="lg" className="navbar-custom">
      <Container>
        <BSNavbar.Brand as={Link} to="/" className="navbar-brand-custom">
          <span className="brand-icon">🕳️</span>
          <span className="brand-text">Pothole Detector AI</span>
        </BSNavbar.Brand>
        <BSNavbar.Toggle aria-controls="basic-navbar-nav" />
        <BSNavbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link
              as={Link}
              to="/"
              className={location.pathname === "/" ? "active" : ""}
            >
              <span className="nav-icon">🔍</span> Detection
            </Nav.Link>
            <Nav.Link
              as={Link}
              to="/ragchat"
              className={location.pathname === "/ragchat" ? "active" : ""}
            >
              <span className="nav-icon">💬</span> RAG Chat
            </Nav.Link>
          </Nav>
        </BSNavbar.Collapse>
      </Container>
    </BSNavbar>
  )
}
