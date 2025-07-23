# Alternative Implementation Ideas

## Option 1: Multiple Backend Support

```python
class RAGPipeline:
    """Main RAG pipeline with optional OpenAI dependency"""
    
    def __init__(self, vector_store: VectorStore, 
                 backend: str = "openrouter", 
                 api_key: str = None,
                 **kwargs):
        self.vector_store = vector_store
        self.backend = backend
        
        if backend == "openrouter":
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
            except ImportError:
                raise ImportError("openai library required for OpenRouter backend")
                
        elif backend == "huggingface":
            try:
                from transformers import pipeline
                self.client = pipeline("text-generation", **kwargs)
            except ImportError:
                raise ImportError("transformers library required for HuggingFace backend")
                
        elif backend == "local":
            # Use a simple template-based response
            self.client = None
            
        else:
            raise ValueError(f"Unsupported backend: {backend}")
```

## Option 2: Abstract Interface

```python
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class OpenRouterBackend(LLMBackend):
    def __init__(self, api_key: str):
        import openai  # Only import when this backend is used
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

class LocalBackend(LLMBackend):
    def generate(self, prompt: str, **kwargs) -> str:
        # Simple keyword-based response (no external dependencies)
        return f"Based on the context, here's a response to: {prompt[:50]}..."
```
