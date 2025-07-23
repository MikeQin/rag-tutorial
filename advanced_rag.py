"""
Advanced RAG Implementation with Reranking and Multi-Modal Support
"""

import base64
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from PIL import Image
from basic_rag import VectorStore, RAGPipeline


class AdvancedRAG(RAGPipeline):
    """Enhanced RAG with reranking capabilities"""
    
    def __init__(self, vector_store: VectorStore, api_key: str):
        super().__init__(vector_store, api_key)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
    
    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        pairs = [(query, doc['text']) for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Combine with original scores
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
            doc['combined_score'] = 0.7 * doc['score'] + 0.3 * doc['rerank_score']
        
        return sorted(documents, key=lambda x: x['combined_score'], reverse=True)
    
    def generate_response_with_reranking(self, query: str, max_tokens: int = 500) -> Dict:
        """Enhanced RAG with reranking"""
        # 1. Initial retrieval (get more candidates)
        relevant_docs = self.vector_store.search(query, n_results=10)
        
        # 2. Rerank documents
        reranked_docs = self.rerank_documents(query, relevant_docs)
        
        # 3. Select top documents after reranking
        top_docs = reranked_docs[:3]
        
        # 4. Generate response
        context = "\n\n".join([
            f"Source: {doc['metadata']['file_name']}\n{doc['text']}"
            for doc in top_docs
        ])
        
        prompt = f"""
        Context information:
        {context}
        
        Based on the context above, please answer the following question:
        {query}
        
        Provide a comprehensive answer and cite specific sources when possible.
        """
        
        response = self.client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # Using OpenRouter format
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed answers based on context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': [doc['metadata']['file_name'] for doc in top_docs],
            'reranking_scores': [doc['combined_score'] for doc in top_docs]
        }


class MultiModalRAG(AdvancedRAG):
    """RAG system with image analysis capabilities"""
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_multimodal_response(self, query: str, image_paths: Optional[List[str]] = None) -> Dict:
        """Generate response with text and image context"""
        # Get text context
        relevant_docs = self.vector_store.search(query, n_results=3)
        text_context = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can analyze both text and images."}
        ]
        
        # Add text context
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Context: {text_context}\n\nQuestion: {query}"
                }
            ]
        }
        
        # Add images if provided
        if image_paths:
            for image_path in image_paths:
                base64_image = self.encode_image(image_path)
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        messages.append(user_message)
        
        response = self.client.chat.completions.create(
            model="openai/gpt-4-vision-preview",  # Using OpenRouter format
            messages=messages,
            max_tokens=500
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': [doc['metadata']['file_name'] for doc in relevant_docs],
            'images_analyzed': len(image_paths) if image_paths else 0
        }


class RAGEvaluator:
    """Evaluation tools for RAG systems"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.embedding_model = rag_system.vector_store.embedding_model
    
    def evaluate_retrieval_relevance(self, queries: List[str], ground_truth_docs: List[List[str]]) -> float:
        """Evaluate retrieval relevance using precision@k"""
        total_precision = 0
        
        for query, truth_docs in zip(queries, ground_truth_docs):
            retrieved_docs = self.rag_system.vector_store.search(query, n_results=5)
            retrieved_doc_ids = [doc['metadata']['file_name'] for doc in retrieved_docs]
            
            relevant_retrieved = len(set(retrieved_doc_ids) & set(truth_docs))
            precision = relevant_retrieved / len(retrieved_doc_ids) if retrieved_doc_ids else 0
            total_precision += precision
        
        return total_precision / len(queries)
    
    def evaluate_answer_quality(self, queries: List[str], ground_truth_answers: List[str]) -> Dict:
        """Evaluate answer quality using semantic similarity"""
        similarities = []
        
        for query, truth_answer in zip(queries, ground_truth_answers):
            generated_answer = self.rag_system.generate_response(query)['answer']
            
            # Calculate semantic similarity
            embeddings = self.embedding_model.encode([generated_answer, truth_answer])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            similarities.append(similarity)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def evaluate_hallucination(self, queries: List[str]) -> float:
        """Simple hallucination detection"""
        hallucination_count = 0
        
        for query in queries:
            result = self.rag_system.generate_response(query)
            answer = result['answer'].lower()
            
            # Check for phrases indicating uncertainty
            uncertainty_phrases = [
                "i don't know",
                "cannot be found in the context",
                "not mentioned in the provided context",
                "insufficient information"
            ]
            
            if not any(phrase in answer for phrase in uncertainty_phrases):
                # Check if answer contains information not in context
                context = result['context_used'].lower()
                answer_embedding = self.embedding_model.encode([answer])
                context_embedding = self.embedding_model.encode([context])
                
                similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
                if similarity < 0.5:  # Threshold for potential hallucination
                    hallucination_count += 1
        
        return hallucination_count / len(queries)


def main():
    """Example usage of advanced RAG features"""
    from basic_rag import DocumentProcessor, VectorStore
    
    # Setup
    processor = DocumentProcessor()
    vector_store = VectorStore("advanced_rag_demo")
    
    # Process documents
    documents = processor.process_documents([
        'sample_documents/manual.txt',
        'sample_documents/faq.txt'
    ])
    vector_store.add_documents(documents)
    
    # Initialize advanced RAG
    advanced_rag = AdvancedRAG(vector_store, "your-openrouter-api-key")
    
    # Test reranking
    query = "What is the return policy?"
    result = advanced_rag.generate_response_with_reranking(query)
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Reranking scores: {result['reranking_scores']}")
    
    # Evaluate system
    evaluator = RAGEvaluator(advanced_rag)
    test_queries = ["What is the return policy?", "How do I contact support?"]
    ground_truth = [["policy.pdf"], ["faq.txt"]]
    
    precision = evaluator.evaluate_retrieval_relevance(test_queries, ground_truth)
    print(f"\nRetrieval Precision@5: {precision:.3f}")


if __name__ == "__main__":
    main()
