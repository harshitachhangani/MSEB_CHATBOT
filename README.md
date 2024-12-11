# MSEB Chatbot 🤖⚡

## Overview

The MSEB Chatbot is an intelligent conversational agent designed to assist users with queries related to the Maharashtra State Electricity Board (MSEB). Leveraging advanced natural language processing (NLP) and machine learning techniques, the chatbot provides quick, accurate, and efficient responses to user inquiries.

## 🎯 Project Motivation

### Why We Created This Chatbot
- **Enhanced User Experience**: Provide instant, accurate responses
- **Automation**: Reduce workload on human support agents
- **Scalability**: Handle multiple queries simultaneously
- **Accessibility**: Potential for website integration

## 🛠 Technologies Stack

### Programming Language
- Python

### Natural Language Processing
- **spaCy**: Text preprocessing
- **NLTK (Natural Language Toolkit)**: Advanced NLP functionalities

### Machine Learning
- **Scikit-learn**: 
  - TF-IDF Vectorization
  - Cosine Similarity Calculations

### Data Handling
- **JSON**: Knowledge base storage
- **BeautifulSoup**: Web scraping for knowledge base enrichment

### Data Visualization
- **Matplotlib**: Performance metrics visualization

## 🧠 Key Components

### Knowledge Base
- Stored in `kb1.json`
- Comprehensive question-answer pairs
- Dynamically expandable

### Text Processing Techniques
- **Tokenization**: Breaking text into meaningful units
- **Lemmatization**: Reducing words to root forms

### Similarity Matching
- **TF-IDF Vectorization**: Convert text to numerical vectors
- **Cosine Similarity**: Identify closest matching queries

## 📊 Performance Metrics

- **Accuracy**: Proportion of correct responses
- **Precision**: Relevance of retrieved responses
- **Recall**: Successfully retrieved relevant responses
- **F1 Score**: Harmonic mean of precision and recall

## 🔧 System Architecture

1. User Query Input
2. Text Preprocessing
3. Vectorization
4. Similarity Calculation
5. Response Generation
6. Optional Web Scraping for Knowledge Base Expansion

## 🚀 Future Enhancements

- Advanced NLP models (BERT, GPT)
- Voice interface implementation
- Real-time learning capabilities
- MSEB website integration
- Multi-language support

## 📦 Installation

### Prerequisites
- Python 3.7+
- pip

### Dependencies
```bash
pip install spacy nltk scikit-learn matplotlib beautifulsoup4
```

### NLTK Data Setup
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 🏃 Running the Project

1. Clone the repository
2. Install dependencies
3. Run the main chatbot script

```bash
git clone https://github.com/yourusername/mseb-chatbot.git
cd mseb-chatbot
pip install -r requirements.txt
python chatbot.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


---

**Crafted with ❤️ by MSEB Chatbot Team**

*Powering Communication, Illuminating Solutions*
