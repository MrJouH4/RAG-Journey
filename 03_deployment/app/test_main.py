import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint returns the correct online status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "Welcome to the RAG Journey Retrieval API"}

def test_ask_endpoint_without_initialization(monkeypatch):
    """Test that the API gracefully handles requests when components aren't ready."""
    # Force vector_store to be None to simulate an uninitialized state
    monkeypatch.setattr("app.main.vector_store", None)
    
    response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 503
    assert "not fully initialized" in response.json()["detail"]