"""
Real-World RAG Applications
This module demonstrates practical RAG implementations for common use cases.
"""

import os
from typing import Dict, List, Optional
from basic_rag import DocumentProcessor, VectorStore, RAGPipeline
from advanced_rag import AdvancedRAG


class CustomerSupportRAG(AdvancedRAG):
    """RAG system specifically designed for customer support"""
    
    def __init__(self, vector_store: VectorStore, api_key: str):
        super().__init__(vector_store, api_key)
        self.escalation_keywords = [
            'urgent', 'complaint', 'refund', 'manager', 'supervisor',
            'cancel', 'angry', 'frustrated', 'legal', 'lawsuit'
        ]
        self.category_keywords = {
            'billing': ['payment', 'charge', 'bill', 'invoice', 'cost'],
            'technical': ['error', 'bug', 'crash', 'not working', 'broken'],
            'account': ['login', 'password', 'access', 'account', 'profile'],
            'shipping': ['delivery', 'shipping', 'tracking', 'package']
        }
    
    def categorize_query(self, query: str) -> str:
        """Categorize customer query"""
        query_lower = query.lower()
        for category, keywords in self.category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return 'general'
    
    def detect_sentiment(self, query: str) -> str:
        """Simple sentiment detection"""
        negative_words = ['angry', 'frustrated', 'terrible', 'awful', 'worst', 'hate']
        positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'wonderful']
        
        query_lower = query.lower()
        negative_count = sum(1 for word in negative_words if word in query_lower)
        positive_count = sum(1 for word in positive_words if word in query_lower)
        
        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'
    
    def handle_support_query(self, query: str, customer_id: Optional[str] = None) -> Dict:
        """Handle customer support queries with enhanced metadata"""
        # Analyze query
        category = self.categorize_query(query)
        sentiment = self.detect_sentiment(query)
        needs_escalation = any(keyword in query.lower() for keyword in self.escalation_keywords)
        
        # Generate response
        result = self.generate_response_with_reranking(query)
        
        # Add support-specific metadata
        result.update({
            'customer_id': customer_id,
            'category': category,
            'sentiment': sentiment,
            'needs_escalation': needs_escalation,
            'confidence': min(result['reranking_scores']) if result['reranking_scores'] else 0,
            'response_type': 'escalation_required' if needs_escalation else 'standard_response'
        })
        
        # Modify response based on sentiment
        if sentiment == 'negative' and not needs_escalation:
            result['answer'] = f"I understand your frustration. {result['answer']}\n\nIs there anything else I can help clarify?"
        
        return result


class DocumentQASystem:
    """Complete document Q&A system with upload capabilities"""
    
    def __init__(self, api_key: str, storage_path: str = "./document_qa_storage"):
        self.storage_path = storage_path
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore("document_qa")
        self.rag = AdvancedRAG(self.vector_store, api_key)
        self.document_registry = {}
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
    
    def upload_document(self, file_path: str, document_type: str = "general") -> Dict:
        """Upload and process a new document"""
        try:
            # Process document
            documents = self.processor.process_documents([file_path])
            
            # Add document type to metadata
            for doc in documents:
                doc['metadata']['document_type'] = document_type
                doc['metadata']['upload_timestamp'] = str(os.path.getmtime(file_path))
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Register document
            file_name = os.path.basename(file_path)
            self.document_registry[file_name] = {
                'file_path': file_path,
                'document_type': document_type,
                'chunks_created': len(documents),
                'upload_timestamp': documents[0]['metadata']['upload_timestamp']
            }
            
            return {
                'status': 'success',
                'file_name': file_name,
                'chunks_created': len(documents),
                'document_type': document_type
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'file_name': os.path.basename(file_path)
            }
    
    def query_documents(self, question: str, document_type: Optional[str] = None) -> Dict:
        """Query across uploaded documents with optional filtering"""
        # If document type specified, modify query to include filter context
        if document_type:
            enhanced_question = f"[Document Type: {document_type}] {question}"
        else:
            enhanced_question = question
        
        result = self.rag.generate_response_with_reranking(enhanced_question)
        
        # Add document registry information
        relevant_docs = []
        for source in result['sources']:
            if source in self.document_registry:
                relevant_docs.append(self.document_registry[source])
        
        result['document_details'] = relevant_docs
        result['query_type'] = document_type or 'general'
        
        return result
    
    def list_documents(self) -> Dict:
        """List all uploaded documents"""
        return {
            'total_documents': len(self.document_registry),
            'documents': self.document_registry
        }


class KnowledgeManagementRAG:
    """Multi-domain knowledge management system"""
    
    def __init__(self, api_key: str):
        self.systems = {
            'hr_policies': VectorStore("hr_policies"),
            'technical_docs': VectorStore("technical_docs"),
            'procedures': VectorStore("procedures"),
            'general': VectorStore("general_knowledge")
        }
        self.api_key = api_key
        self.routing_keywords = {
            'hr_policies': ['vacation', 'leave', 'policy', 'hr', 'employee', 'benefit', 'handbook'],
            'technical_docs': ['api', 'code', 'technical', 'system', 'software', 'development', 'bug'],
            'procedures': ['process', 'procedure', 'workflow', 'step', 'guide', 'instruction']
        }
    
    def route_query(self, query: str) -> str:
        """Route query to appropriate knowledge base using keyword matching"""
        query_lower = query.lower()
        scores = {}
        
        for domain, keywords in self.routing_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[domain] = score
        
        # Return domain with highest score, or 'general' if no matches
        best_domain = max(scores.items(), key=lambda x: x[1])
        return best_domain[0] if best_domain[1] > 0 else 'general'
    
    def add_documents_to_domain(self, domain: str, documents: List[Dict]) -> Dict:
        """Add documents to a specific knowledge domain"""
        if domain not in self.systems:
            return {'status': 'error', 'message': f'Unknown domain: {domain}'}
        
        self.systems[domain].add_documents(documents)
        return {
            'status': 'success',
            'domain': domain,
            'documents_added': len(documents)
        }
    
    def query_knowledge_base(self, query: str, force_domain: Optional[str] = None) -> Dict:
        """Query the appropriate knowledge base"""
        # Determine which system to use
        if force_domain and force_domain in self.systems:
            system_type = force_domain
        else:
            system_type = self.route_query(query)
        
        vector_store = self.systems[system_type]
        rag = AdvancedRAG(vector_store, self.api_key)
        
        result = rag.generate_response_with_reranking(query)
        result['knowledge_domain'] = system_type
        result['routing_confidence'] = 'forced' if force_domain else 'automatic'
        
        return result
    
    def cross_domain_search(self, query: str, max_results_per_domain: int = 2) -> Dict:
        """Search across all knowledge domains"""
        all_results = {}
        combined_sources = []
        
        for domain, vector_store in self.systems.items():
            try:
                domain_results = vector_store.search(query, n_results=max_results_per_domain)
                all_results[domain] = domain_results
                combined_sources.extend([f"{domain}:{result['metadata']['file_name']}" for result in domain_results])
            except Exception as e:
                all_results[domain] = {'error': str(e)}
        
        return {
            'query': query,
            'cross_domain_results': all_results,
            'combined_sources': combined_sources,
            'domains_searched': list(self.systems.keys())
        }


def main():
    """Example usage of real-world RAG applications"""
    # Customer Support RAG Example
    print("=== Customer Support RAG Demo ===")
    
    # Setup
    processor = DocumentProcessor()
    support_vector_store = VectorStore("customer_support")
    
    # Process support documents
    support_docs = processor.process_documents([
        'sample_documents/faq.txt',
        'sample_documents/return_policy.txt',
        'sample_documents/troubleshooting.txt'
    ])
    support_vector_store.add_documents(support_docs)
    
    # Initialize customer support RAG
    support_rag = CustomerSupportRAG(support_vector_store, "your-openrouter-api-key")
    
    # Test customer queries
    test_queries = [
        "I'm angry! My order is late and I want a refund!",
        "How do I reset my password?",
        "What's your return policy for electronics?"
    ]
    
    for query in test_queries:
        result = support_rag.handle_support_query(query, customer_id="CUST123")
        print(f"\nQuery: {query}")
        print(f"Category: {result['category']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Needs Escalation: {result['needs_escalation']}")
        print(f"Answer: {result['answer'][:100]}...")
        print("-" * 50)
    
    
    # Document Q&A System Example
    print("\n=== Document Q&A System Demo ===")
    
    qa_system = DocumentQASystem("your-openrouter-api-key")
    
    # Upload documents
    upload_results = [
        qa_system.upload_document('sample_documents/manual.pdf', 'technical'),
        qa_system.upload_document('sample_documents/policies.txt', 'policy')
    ]
    
    for result in upload_results:
        print(f"Upload result: {result}")
    
    # Query documents
    query_result = qa_system.query_documents("How do I install the software?", document_type="technical")
    print(f"\nQuery result: {query_result['answer'][:100]}...")
    
    # List documents
    doc_list = qa_system.list_documents()
    print(f"Total documents: {doc_list['total_documents']}")


if __name__ == "__main__":
    main()
