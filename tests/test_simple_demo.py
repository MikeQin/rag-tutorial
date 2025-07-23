"""
Tests for simple_demo.py
"""
import pytest
from simple_demo import SimpleRAGDemo, create_sample_knowledge_base


class TestSimpleRAGDemo:
    """Test the simple RAG demo functionality"""
    
    def test_add_document(self):
        """Test adding documents to the demo"""
        rag = SimpleRAGDemo()
        doc_id = rag.add_document("This is a test document", "Test Doc")
        
        assert doc_id == 0
        assert len(rag.documents) == 1
        assert rag.documents[0]['title'] == "Test Doc"
        assert rag.documents[0]['text'] == "This is a test document"
    
    def test_simple_search(self):
        """Test the simple keyword search"""
        rag = SimpleRAGDemo()
        rag.add_document("This document is about machine learning", "ML Doc")
        rag.add_document("This document is about cooking recipes", "Recipe Doc")
        
        results = rag.simple_search("machine learning", n_results=2)
        
        assert len(results) >= 1
        assert results[0]['document']['title'] == "ML Doc"
        assert results[0]['score'] > 0
    
    def test_simple_search_no_results(self):
        """Test search with no matching documents"""
        rag = SimpleRAGDemo()
        rag.add_document("This is about cats", "Cat Doc")
        
        results = rag.simple_search("dogs", n_results=2)
        
        assert len(results) == 0
    
    def test_generate_simple_response(self):
        """Test response generation"""
        rag = SimpleRAGDemo()
        rag.add_document("Our return policy allows returns within 30 days", "Policy")
        
        result = rag.generate_simple_response("What is the return policy?")
        
        assert 'answer' in result
        assert 'sources' in result
        assert 'context' in result
        assert len(result['sources']) > 0
        assert "Policy" in result['sources']
    
    def test_generate_response_no_context(self):
        """Test response when no relevant documents found"""
        rag = SimpleRAGDemo()
        rag.add_document("This is about cats", "Cat Doc")
        
        result = rag.generate_simple_response("What about dogs?")
        
        assert "couldn't find any relevant information" in result['answer']
        assert len(result['sources']) == 0


class TestSampleKnowledgeBase:
    """Test the sample knowledge base creation"""
    
    def test_create_sample_knowledge_base(self):
        """Test creating the sample knowledge base"""
        rag = create_sample_knowledge_base()
        
        assert len(rag.documents) == 5
        
        # Check that we have the expected documents
        titles = [doc['title'] for doc in rag.documents]
        expected_titles = [
            'Return Policy', 
            'Shipping Information', 
            'Customer Support',
            'Account Management',
            'Product Warranty'
        ]
        
        for title in expected_titles:
            assert title in titles
    
    def test_sample_queries(self):
        """Test some sample queries on the knowledge base"""
        rag = create_sample_knowledge_base()
        
        # Test return policy query
        result = rag.generate_simple_response("What is your return policy?")
        assert len(result['sources']) > 0
        assert any('Return Policy' in source for source in result['sources'])
        
        # Test shipping query
        result = rag.generate_simple_response("Do you offer free shipping?")
        assert len(result['sources']) > 0
        assert any('Shipping' in source for source in result['sources'])


if __name__ == "__main__":
    pytest.main([__file__])
