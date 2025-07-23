"""
Basic RAG Implementation
This module provides a complete, working RAG system implementation.
"""

import os
from typing import List, Dict
import openai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
import tiktoken


class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> List[Dict]:
        """Process multiple documents into chunks with metadata"""
        processed_docs = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                processed_docs.append({
                    'text': chunk,
                    'source': file_path,
                    'chunk_id': i,
                    'metadata': {
                        'file_name': os.path.basename(file_path),
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                })
        
        return processed_docs


class VectorStore:
    """Handles vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.get_collection(name=collection_name)
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store"""
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        ids = [f"{doc['source']}_{doc['chunk_id']}" for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return [{
            'text': doc,
            'metadata': meta,
            'score': 1 - dist  # Convert distance to similarity score
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )]


class RAGPipeline:
    """Main RAG pipeline implementation"""
    
    def __init__(self, vector_store: VectorStore, api_key: str):
        self.vector_store = vector_store
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def generate_response(self, query: str, max_tokens: int = 500) -> Dict:
        """Generate response using RAG"""
        # 1. Retrieve relevant documents
        relevant_docs = self.vector_store.search(query, n_results=3)
        
        # 2. Prepare context
        context = "\n\n".join([
            f"Source: {doc['metadata']['file_name']}\n{doc['text']}"
            for doc in relevant_docs
        ])
        
        # 3. Create prompt
        prompt = f"""
        Context information:
        {context}
        
        Based on the context above, please answer the following question:
        {query}
        
        If the answer cannot be found in the context, please say so clearly.
        """
        
        # 4. Generate response
        response = self.client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # Using OpenRouter format
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': [doc['metadata']['file_name'] for doc in relevant_docs],
            'context_used': context,
            'relevance_scores': [doc['score'] for doc in relevant_docs]
        }


def main():
    """Example usage of the RAG system"""
    # Initialize components
    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Process documents (add your document paths here)
    documents = processor.process_documents([
        'sample_documents/manual.txt',
        'sample_documents/faq.txt'
    ])
    
    # Add to vector store
    vector_store.add_documents(documents)
    
    # Initialize RAG pipeline (add your OpenRouter API key)
    rag = RAGPipeline(vector_store, "your-openrouter-api-key")
    
    # Test queries
    test_queries = [
        "What is the return policy?",
        "How do I contact customer support?",
        "What are the warranty terms?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.generate_response(query)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
