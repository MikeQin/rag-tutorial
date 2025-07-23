# 🤖 RAG Tutorial: Complete Guide to Retrieval Augmented Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenRouter](https://img.shields.io/badge/API-OpenRouter-green.svg)](https://openrouter.ai/)

A comprehensive, hands-on tutorial for building production-ready Retrieval Augmented Generation (RAG) systems. Learn from basic concepts to advanced implementations with real-world examples.

## 🚀 Quick Start

1. **Run the Simple Demo (No Dependencies)**:
   ```bash
   python simple_demo.py
   ```

2. **Install Full Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Your OpenRouter API Key**:
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```
   
   Get your API key from [OpenRouter](https://openrouter.ai/keys)

4. **Try the Full RAG System**:
   ```bash
   python basic_rag.py
   ```

## 📁 File Structure

```
rag-tutorial/
├── 📄 README.md                 # You are here!
├── 📄 LICENSE                   # MIT License
├── 📄 CONTRIBUTING.md           # Contribution guidelines
├── 📄 .gitignore               # Git ignore rules
├── 📄 requirements.txt          # Python dependencies
├── 📄 requirements-dev.txt      # Development dependencies
├── 📚 rag-tutorial.md          # Complete tutorial documentation
├── ⚙️  openrouter-setup.md      # OpenRouter configuration guide
├── 🎯 simple_demo.py           # No-dependency demo (start here!)
├── 🔧 basic_rag.py            # Full RAG implementation
├── ⚡ advanced_rag.py         # Enhanced RAG with reranking
├── 🔍 hybrid_search.py        # Keyword + semantic search
├── 🏭 applications.py         # Real-world examples
├── 📁 sample_documents/       # Example documents
│   ├── 📄 manual.txt
│   ├── 📄 faq.txt
│   └── 📄 policies.txt
├── 📁 tests/                  # Unit tests (coming soon)
└── 📁 chroma_db/             # Vector database (auto-created)
```

## 🎓 Learning Path

1. **📖 Understand Concepts**: Read [`rag-tutorial.md`](rag-tutorial.md)
2. **👀 See It Work**: Run [`simple_demo.py`](simple_demo.py) 
3. **🔧 Try Real RAG**: Use [`basic_rag.py`](basic_rag.py)
4. **⚡ Advanced Features**: Experiment with [`advanced_rag.py`](advanced_rag.py)
5. **🏭 Production Ready**: Study [`applications.py`](applications.py)

## 🔧 Features

- ✅ **Progressive Learning**: Start simple, build complexity
- ✅ **Working Examples**: All code is tested and functional
- ✅ **Multiple Models**: OpenRouter integration for model variety
- ✅ **Real Applications**: Customer support, document Q&A, knowledge management
- ✅ **Advanced Techniques**: Reranking, hybrid search, multi-modal RAG
- ✅ **Evaluation Tools**: Metrics and testing frameworks
- ✅ **Production Ready**: Error handling, logging, configuration

## Sample Documents

Create a `sample_documents/` folder and add some text files:

**faq.txt**:
```
Q: What is your return policy?
A: We accept returns within 30 days of purchase with original receipt.

Q: How do I contact support?
A: Email us at support@company.com or call 1-800-555-0123.
```

**manual.txt**:
```
Product Installation Guide

1. Download the software from our website
2. Run the installer as administrator
3. Follow the setup wizard
4. Restart your computer when prompted
```

## Common Issues

1. **Import Errors**: Install requirements with `pip install -r requirements.txt`
2. **API Key Missing**: Set your OpenRouter API key in environment variables
3. **ChromaDB Issues**: Delete `chroma_db/` folder and restart
4. **Memory Issues**: Reduce chunk size in DocumentProcessor

## Next Steps

- Try with your own documents
- Experiment with different embedding models
- Implement custom evaluation metrics
- Deploy as a web service with FastAPI

Happy RAG building! 🚀

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- [OpenRouter](https://openrouter.ai/) for accessible AI model APIs
- [ChromaDB](https://www.trychroma.com/) for vector database functionality
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- The open-source AI community for inspiration and tools

## 📞 Support

- 🐛 **Bug Reports**: [Create an issue](https://github.com/your-username/rag-tutorial/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/your-username/rag-tutorial/discussions)
- ❓ **Questions**: Check existing issues or start a new discussion

---

⭐ **Found this tutorial helpful? Give it a star!** ⭐
