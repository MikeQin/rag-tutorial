"""
Test configuration and fixtures for RAG Tutorial tests
"""
import os
import tempfile
import pytest
from unittest.mock import Mock


@pytest.fixture
def sample_text():
    """Sample text for testing document processing"""
    return """
    This is a sample document for testing purposes.
    It contains multiple sentences and paragraphs.
    
    The document discusses various topics including:
    - Natural language processing
    - Machine learning
    - Artificial intelligence
    
    This text will be used to test chunking and embedding functionality.
    """


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            'text': 'This is the first test document about machine learning.',
            'source': 'doc1.txt',
            'chunk_id': 0,
            'metadata': {'file_name': 'doc1.txt', 'chunk_index': 0, 'total_chunks': 1}
        },
        {
            'text': 'This is the second test document about artificial intelligence.',
            'source': 'doc2.txt', 
            'chunk_id': 0,
            'metadata': {'file_name': 'doc2.txt', 'chunk_index': 0, 'total_chunks': 1}
        }
    ]


@pytest.fixture
def temp_file():
    """Create a temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is a test file content.\nWith multiple lines.\nFor testing purposes.")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a test response from the mock LLM."
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing"""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return mock_model


@pytest.fixture
def skip_if_no_api_key():
    """Skip test if no API key is available"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your-openrouter-api-key":
        pytest.skip("No valid OpenRouter API key available for integration tests")
    return api_key
