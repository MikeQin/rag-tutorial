"""
Hybrid Search Implementation
Combines semantic search with keyword-based search (BM25)
"""

from typing import List, Dict
from rank_bm25 import BM25Okapi
from basic_rag import VectorStore


class HybridSearch:
    """Combines semantic and keyword search for better retrieval"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
    
    def index_documents(self, documents: List[Dict]):
        """Index documents for BM25 search"""
        self.documents = [doc['text'] for doc in documents]
        self.doc_metadata = documents
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = 0.7) -> List[Dict]:
        """Combine semantic and keyword search"""
        if not self.bm25:
            # Fallback to semantic search only
            return self.vector_store.search(query, n_results=n_results)
        
        # Semantic search
        semantic_results = self.vector_store.search(query, n_results=n_results*2)
        
        # Keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Create mapping from document text to BM25 score
        doc_to_bm25 = {doc: score for doc, score in zip(self.documents, bm25_scores)}
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if bm25_scores else 1
        
        # Combine scores
        combined_results = []
        for doc_result in semantic_results:
            semantic_score = doc_result['score']
            bm25_score = doc_to_bm25.get(doc_result['text'], 0)
            normalized_bm25 = bm25_score / (max_bm25 + 1e-6)
            
            combined_score = alpha * semantic_score + (1 - alpha) * normalized_bm25
            
            doc_result['combined_score'] = combined_score
            doc_result['bm25_score'] = normalized_bm25
            combined_results.append(doc_result)
        
        # Sort by combined score and return top results
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined_results[:n_results]


class QueryEnhancer:
    """Enhance queries for better retrieval"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-3.5-turbo",  # Using OpenRouter format
                messages=[
                    {"role": "system", "content": "Generate 3-5 related search terms for the given query. Return only the terms, separated by commas."},
                    {"role": "user", "content": query}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            expanded_terms = response.choices[0].message.content
            return f"{query} {expanded_terms}"
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query
    
    def rephrase_query(self, query: str) -> List[str]:
        """Generate multiple phrasings of the same query"""
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-3.5-turbo",  # Using OpenRouter format
                messages=[
                    {"role": "system", "content": "Rephrase the following query in 3 different ways. Return each rephrasing on a new line."},
                    {"role": "user", "content": query}
                ],
                max_tokens=100,
                temperature=0.5
            )
            
            rephrasings = response.choices[0].message.content.strip().split('\n')
            return [query] + [r.strip() for r in rephrasings if r.strip()]
        except Exception as e:
            print(f"Query rephrasing failed: {e}")
            return [query]


class DocumentPreprocessor:
    """Advanced document preprocessing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\'\"()]', '', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(lines)
    
    @staticmethod
    def add_metadata_context(chunk: str, metadata: Dict) -> str:
        """Add metadata context to chunks"""
        context_header = f"Document: {metadata.get('file_name', 'Unknown')}\n"
        if 'section' in metadata:
            context_header += f"Section: {metadata['section']}\n"
        context_header += "\n"
        return context_header + chunk
    
    @staticmethod
    def extract_structure(text: str) -> Dict:
        """Extract document structure (headers, sections, etc.)"""
        import re
        
        # Find potential headers (lines that start with #, are all caps, or are short and end with :)
        lines = text.split('\n')
        headers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if (line.startswith('#') or 
                (line.isupper() and len(line.split()) <= 5) or
                (len(line) <= 50 and line.endswith(':'))):
                headers.append({
                    'text': line,
                    'line_number': i,
                    'level': line.count('#') if line.startswith('#') else 1
                })
        
        return {
            'headers': headers,
            'total_lines': len(lines),
            'estimated_sections': len(headers)
        }


def main():
    """Example usage of hybrid search and query enhancement"""
    from basic_rag import DocumentProcessor, VectorStore, RAGPipeline
    import openai
    
    # Setup
    processor = DocumentProcessor()
    vector_store = VectorStore("hybrid_search_demo")
    
    # Process documents
    documents = processor.process_documents([
        'sample_documents/manual.txt',
        'sample_documents/faq.txt'
    ])
    vector_store.add_documents(documents)
    
    # Setup hybrid search
    hybrid_search = HybridSearch(vector_store)
    hybrid_search.index_documents(documents)
    
    # Setup query enhancer
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="your-openrouter-api-key"
    )
    query_enhancer = QueryEnhancer(client)
    
    # Test queries
    query = "return policy"
    
    # Regular semantic search
    semantic_results = vector_store.search(query, n_results=3)
    print("Semantic Search Results:")
    for i, result in enumerate(semantic_results):
        print(f"{i+1}. Score: {result['score']:.3f} - {result['text'][:100]}...")
    
    # Hybrid search
    hybrid_results = hybrid_search.hybrid_search(query, n_results=3)
    print("\nHybrid Search Results:")
    for i, result in enumerate(hybrid_results):
        print(f"{i+1}. Combined Score: {result['combined_score']:.3f}, BM25: {result['bm25_score']:.3f} - {result['text'][:100]}...")
    
    # Query expansion
    expanded_query = query_enhancer.expand_query(query)
    print(f"\nOriginal query: {query}")
    print(f"Expanded query: {expanded_query}")
    
    # Query rephrasing
    rephrased_queries = query_enhancer.rephrase_query(query)
    print(f"\nRephrased queries: {rephrased_queries}")


if __name__ == "__main__":
    main()
