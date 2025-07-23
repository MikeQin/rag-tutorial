# Contributing to RAG Tutorial

Thank you for your interest in contributing to the RAG Tutorial! This document provides guidelines for contributing to this project.

## How to Contribute

### 1. Fork and Clone
```bash
git fork https://github.com/your-username/rag-tutorial
git clone https://github.com/your-username/rag-tutorial.git
cd rag-tutorial
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Make Your Changes
- Create a new branch for your feature: `git checkout -b feature/your-feature-name`
- Make your changes
- Test your changes thoroughly
- Follow the coding standards outlined below

### 4. Submit a Pull Request
- Push your changes to your fork
- Create a pull request with a clear description
- Reference any related issues

## Coding Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write clear, descriptive docstrings
- Keep functions focused and small
- Use meaningful variable names

### Documentation
- Update README.md if adding new features
- Add docstrings to all new functions and classes
- Include examples in docstrings when helpful
- Update the main tutorial if adding new concepts

### Testing
- Add tests for new functionality
- Ensure all existing tests pass
- Test with different models and configurations
- Include edge cases in your tests

## Types of Contributions

### 🐛 Bug Reports
- Use the issue template
- Include steps to reproduce
- Provide error messages and stack traces
- Specify your environment (OS, Python version, etc.)

### 💡 Feature Requests
- Describe the problem you're solving
- Explain the proposed solution
- Consider alternative approaches
- Discuss potential breaking changes

### 📚 Documentation Improvements
- Fix typos and grammar
- Clarify confusing sections
- Add missing examples
- Improve code comments

### 🔧 Code Contributions
- New RAG techniques or algorithms
- Performance improvements
- Additional model integrations
- Better error handling
- Code refactoring

## Development Guidelines

### Adding New Models
When adding support for new models:
1. Update the configuration in `openrouter-setup.md`
2. Add model-specific error handling
3. Include cost and performance information
4. Test with different query types
5. Document any model-specific limitations

### Adding New Features
For new RAG features:
1. Add to the appropriate module (basic, advanced, etc.)
2. Include comprehensive docstrings
3. Add usage examples
4. Update the main tutorial document
5. Consider backward compatibility

### Code Organization
- Keep related functionality together
- Use clear module and class names
- Separate concerns appropriately
- Avoid circular dependencies

## Code Review Process

### What We Look For
- Code quality and clarity
- Proper error handling
- Performance considerations
- Security implications
- Documentation completeness
- Test coverage

### Review Timeline
- Initial response: 2-3 days
- Detailed review: 1 week
- Follow-up iterations: 2-3 days each

## Community Guidelines

### Be Respectful
- Use inclusive language
- Respect different perspectives
- Provide constructive feedback
- Help newcomers learn

### Stay on Topic
- Keep discussions relevant to RAG and AI
- Use appropriate channels for different topics
- Search existing issues before creating new ones

### Quality Standards
- Test your code before submitting
- Follow the style guide
- Write clear commit messages
- Keep pull requests focused

## Getting Help

### Documentation
- Read the main tutorial: `rag-tutorial.md`
- Check setup guides: `README.md` and `openrouter-setup.md`
- Review existing code examples

### Community Support
- GitHub Issues for bugs and feature requests
- GitHub Discussions for general questions
- Stack Overflow for specific coding questions (tag: `rag-tutorial`)

### Maintainer Contact
- Create an issue for project-related questions
- Use GitHub Discussions for broader conversations

## Recognition

Contributors will be:
- Listed in the README.md contributors section
- Credited in release notes for significant contributions
- Given appropriate GitHub repository permissions for ongoing contributors

## Development Setup Details

### Environment Variables
```bash
# Required for testing
export OPENROUTER_API_KEY="your-test-key"

# Optional for development
export RAG_DEBUG=true
export RAG_LOG_LEVEL=debug
```

### Testing Commands
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_basic_rag.py

# Run with coverage
python -m pytest --cov=rag_tutorial

# Run linting
flake8 rag_tutorial/
black rag_tutorial/
```

### Pre-commit Hooks
We recommend setting up pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Thank you for contributing to the RAG Tutorial! 🚀
