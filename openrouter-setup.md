# OpenRouter Configuration for RAG Tutorial

## Why OpenRouter?

OpenRouter provides access to multiple AI models through a single API:
- **Cost-effective**: Often cheaper than direct OpenAI API
- **Model variety**: Access to GPT, Claude, Llama, and more
- **Reliability**: Built-in failover and load balancing
- **Flexibility**: Easy model switching without code changes

## Setup Instructions

1. **Get API Key**:
   - Visit [OpenRouter](https://openrouter.ai/)
   - Sign up for an account
   - Go to [API Keys](https://openrouter.ai/keys)
   - Create a new API key

2. **Set Environment Variable**:
   ```bash
   # Windows
   set OPENROUTER_API_KEY=your-key-here
   
   # macOS/Linux
   export OPENROUTER_API_KEY=your-key-here
   
   # Or add to your .env file
   echo "OPENROUTER_API_KEY=your-key-here" >> .env
   ```

3. **Install Additional Package** (optional):
   ```bash
   pip install python-dotenv
   ```

## Model Options

OpenRouter supports many models. Here are some popular choices:

### **For RAG Applications**:
- `openai/gpt-3.5-turbo` - Fast and cost-effective
- `openai/gpt-4` - Best quality, higher cost
- `anthropic/claude-3-haiku` - Good balance of speed/quality
- `meta-llama/llama-2-70b-chat` - Open source alternative

### **For Vision Tasks**:
- `openai/gpt-4-vision-preview` - Best for image analysis
- `anthropic/claude-3-opus` - Excellent multimodal capabilities

### **Cost Optimization**:
- `openai/gpt-3.5-turbo` - $0.50/1M tokens (input), $1.50/1M tokens (output)
- `anthropic/claude-3-haiku` - $0.25/1M tokens (input), $1.25/1M tokens (output)
- `meta-llama/llama-2-70b-chat` - $0.65/1M tokens (input/output)

## Code Configuration

Update your RAG pipeline initialization:

```python
import os
import openai

# Using environment variable
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenAI client for OpenRouter
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Use in your RAG pipeline
rag = RAGPipeline(vector_store, api_key)
```

## Advanced Configuration

### **Model Selection by Use Case**:
```python
class ModelConfig:
    # Fast responses for simple queries
    FAST_MODEL = "openai/gpt-3.5-turbo"
    
    # High quality for complex analysis
    QUALITY_MODEL = "openai/gpt-4"
    
    # Vision tasks
    VISION_MODEL = "openai/gpt-4-vision-preview"
    
    # Cost-optimized
    BUDGET_MODEL = "meta-llama/llama-2-70b-chat"

def get_model_for_query(query: str) -> str:
    if len(query) > 500 or "analyze" in query.lower():
        return ModelConfig.QUALITY_MODEL
    return ModelConfig.FAST_MODEL
```

### **Error Handling and Fallbacks**:
```python
def create_completion_with_fallback(client, messages, primary_model, fallback_model):
    try:
        return client.chat.completions.create(
            model=primary_model,
            messages=messages,
            max_tokens=500
        )
    except Exception as e:
        print(f"Primary model failed: {e}")
        return client.chat.completions.create(
            model=fallback_model,
            messages=messages,
            max_tokens=500
        )
```

### **Usage Tracking**:
```python
import time

class UsageTracker:
    def __init__(self):
        self.usage_log = []
    
    def log_request(self, model, tokens_used, cost=None):
        self.usage_log.append({
            'timestamp': time.time(),
            'model': model,
            'tokens': tokens_used,
            'cost': cost
        })
    
    def get_daily_usage(self):
        today = time.time() - 86400  # 24 hours ago
        return [log for log in self.usage_log if log['timestamp'] > today]
```

## Benefits for RAG

1. **Cost Control**: Switch to cheaper models for simple queries
2. **Performance**: Use faster models for real-time responses
3. **Reliability**: Automatic failover if a model is unavailable
4. **Experimentation**: Easy A/B testing with different models
5. **Future-proofing**: Access to new models as they're released

## Migration from OpenAI

If you're migrating from direct OpenAI API:

1. Change the base URL to OpenRouter
2. Update model names to include provider prefix
3. Keep the same OpenAI client library
4. Optionally add model selection logic

The API is fully compatible, so existing code works with minimal changes!
