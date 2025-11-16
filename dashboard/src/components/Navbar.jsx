import React from 'react';
import { Navbar as BSNavbar, Container, Nav } from 'react-bootstrap';

export default function Navbar() {
  return (
    <BSNavbar variant="dark" expand="lg" className="shadow-sm">
      <Container>
        <BSNavbar.Brand href="#home" className="fw-bold fs-4">
          ML Dashboard
        </BSNavbar.Brand>
        <BSNavbar.Toggle aria-controls="basic-navbar-nav" />
        <BSNavbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link href="#home">Home</Nav.Link>
            <Nav.Link href="#results">Results</Nav.Link>
            <Nav.Link href="#metrics">Metrics</Nav.Link>
          </Nav>
        </BSNavbar.Collapse>
      </Container>
    </BSNavbar>
  );
}
