"""
Simple RAG Demo - Quick Start Example
This is a minimal example to get you started with RAG quickly.
"""

import os
from typing import List, Dict

# For this demo, we'll create a simple in-memory vector store
# In production, use ChromaDB, Pinecone, or similar


class SimpleRAGDemo:
    """Minimal RAG implementation for learning purposes"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = {}
        
    def add_document(self, text: str, title: str = "Document"):
        """Add a document to our simple knowledge base"""
        doc_id = len(self.documents)
        self.documents.append({
            'id': doc_id,
            'title': title,
            'text': text
        })
        return doc_id
    
    def simple_search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Simple keyword-based search (not semantic)"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            doc_words = set(doc['text'].lower().split())
            # Simple overlap score
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                results.append({
                    'document': doc,
                    'score': overlap / len(query_words),
                    'text': doc['text']
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    def generate_simple_response(self, query: str) -> Dict:
        """Generate a simple response without LLM (for demo purposes)"""
        # Search for relevant documents
        relevant_docs = self.simple_search(query, n_results=2)
        
        if not relevant_docs:
            return {
                'answer': "I couldn't find any relevant information in the knowledge base.",
                'sources': [],
                'context': ""
            }
        
        # Combine context
        context = "\n\n".join([doc['text'] for doc in relevant_docs])
        sources = [doc['document']['title'] for doc in relevant_docs]
        
        # Simple response generation (in real RAG, this would use an LLM)
        answer = f"Based on the available information: {context[:200]}..."
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context,
            'relevance_scores': [doc['score'] for doc in relevant_docs]
        }


def create_sample_knowledge_base() -> SimpleRAGDemo:
    """Create a sample knowledge base for demonstration"""
    rag = SimpleRAGDemo()
    
    # Add sample documents
    sample_docs = [
        {
            'title': 'Return Policy',
            'text': 'Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with receipt. Refunds are processed within 5-7 business days. Digital products cannot be returned unless defective.'
        },
        {
            'title': 'Shipping Information',
            'text': 'We offer free shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping is available for $9.99 and takes 1-2 business days. International shipping is available to most countries.'
        },
        {
            'title': 'Customer Support',
            'text': 'Customer support is available Monday-Friday 9AM-6PM EST. You can reach us by phone at 1-800-555-0123, email at support@company.com, or live chat on our website. We typically respond to emails within 24 hours.'
        },
        {
            'title': 'Account Management',
            'text': 'You can manage your account by logging in to our website. From your account dashboard, you can update personal information, view order history, track shipments, and manage payment methods. Password reset is available on the login page.'
        },
        {
            'title': 'Product Warranty',
            'text': 'All products come with a standard 1-year warranty against manufacturing defects. Extended warranty options are available for purchase. Warranty claims can be submitted through your account or by contacting customer support.'
        }
    ]
    
    for doc in sample_docs:
        rag.add_document(doc['text'], doc['title'])
    
    return rag


def demo_rag_queries():
    """Demonstrate RAG with sample queries"""
    print("🚀 Simple RAG Demo")
    print("=" * 50)
    
    # Create knowledge base
    rag = create_sample_knowledge_base()
    
    # Test queries
    test_queries = [
        "What is your return policy?",
        "How can I contact customer support?",
        "Do you offer free shipping?",
        "How do I reset my password?",
        "What warranty do you provide?",
        "Can I return digital products?"  # This should show partial matching
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = rag.generate_simple_response(query)
        
        print(f"📋 Sources: {', '.join(result['sources'])}")
        print(f"⭐ Relevance Scores: {[f'{score:.2f}' for score in result['relevance_scores']]}")
        print(f"💬 Answer: {result['answer']}")
        print("-" * 50)


def explain_rag_concepts():
    """Explain key RAG concepts with the demo"""
    print("\n🎓 RAG Concepts Explained")
    print("=" * 50)
    
    concepts = [
        {
            'concept': 'Knowledge Base',
            'explanation': 'A collection of documents that the system can search through. In our demo, we have 5 documents about company policies.'
        },
        {
            'concept': 'Retrieval',
            'explanation': 'Finding relevant documents based on the user query. Our simple demo uses keyword matching, but real RAG uses semantic similarity.'
        },
        {
            'concept': 'Context Assembly',
            'explanation': 'Combining the retrieved documents into context for the language model. We concatenate the relevant document texts.'
        },
        {
            'concept': 'Generation',
            'explanation': 'Using an LLM to generate a response based on the retrieved context. Our demo simplifies this step for illustration.'
        },
        {
            'concept': 'Relevance Scoring',
            'explanation': 'Measuring how relevant each document is to the query. Higher scores mean better matches.'
        }
    ]
    
    for concept in concepts:
        print(f"\n📚 {concept['concept']}:")
        print(f"   {concept['explanation']}")


def upgrade_path():
    """Show how to upgrade from simple demo to production RAG"""
    print("\n🔧 Upgrading to Production RAG")
    print("=" * 50)
    
    upgrades = [
        {
            'component': 'Search Method',
            'current': 'Simple keyword matching',
            'upgrade_to': 'Semantic search with embeddings (sentence-transformers)',
            'benefit': 'Better understanding of query meaning'
        },
        {
            'component': 'Vector Storage',
            'current': 'In-memory list',
            'upgrade_to': 'ChromaDB, Pinecone, or Weaviate',
            'benefit': 'Scalable, persistent, optimized search'
        },
        {
            'component': 'Response Generation',
            'current': 'Simple text combination',
            'upgrade_to': 'OpenAI GPT, Claude, or local LLM',
            'benefit': 'Natural, contextual responses'
        },
        {
            'component': 'Document Processing',
            'current': 'Manual text input',
            'upgrade_to': 'PDF extraction, chunking, metadata',
            'benefit': 'Handle real documents automatically'
        },
        {
            'component': 'Evaluation',
            'current': 'Manual testing',
            'upgrade_to': 'Automated metrics (precision, recall, relevance)',
            'benefit': 'Measure and improve system performance'
        }
    ]
    
    for upgrade in upgrades:
        print(f"\n🔄 {upgrade['component']}:")
        print(f"   Current: {upgrade['current']}")
        print(f"   Upgrade to: {upgrade['upgrade_to']}")
        print(f"   Benefit: {upgrade['benefit']}")


def main():
    """Run the complete RAG demo with explanations"""
    # Run the demo
    demo_rag_queries()
    
    # Explain concepts
    explain_rag_concepts()
    
    # Show upgrade path
    upgrade_path()
    
    print("\n🎯 Next Steps:")
    print("1. Install the full requirements: pip install -r requirements.txt")
    print("2. Try the basic_rag.py example with real embeddings")
    print("3. Experiment with advanced_rag.py for reranking")
    print("4. Build your own RAG system with your documents!")


if __name__ == "__main__":
    main()
