# MSEB Chatbot Project

## Introduction

This project involves developing a chatbot designed to assist users with queries related to the Maharashtra State Electricity Board (MSEB). By leveraging natural language processing (NLP) techniques and machine learning algorithms, the chatbot can understand user queries and provide relevant responses based on a pre-defined knowledge base. 

## Why We Chose This Project

1. **Enhanced User Experience**: To provide quick and accurate responses to user queries, improving customer satisfaction.
2. **Automation**: To automate the process of handling common queries, reducing the workload on human support agents.
3. **Scalability**: To handle a large volume of queries simultaneously without compromising response quality.
4. **Integration Potential**: The chatbot can be integrated with the actual MSEB website, enhancing its utility and accessibility.

## Technologies Used

### Programming Language
- **Python**: The primary programming language used for implementing the chatbot and associated functionalities.

### Natural Language Processing (NLP)
- **spaCy**: Utilized for text preprocessing tasks such as tokenization and lemmatization.
- **NLTK (Natural Language Toolkit)**: Provides additional NLP functionalities, including word tokenization and lemmatization.

### Machine Learning
- **Scikit-learn**: Used for vectorization tasks with the TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer and for computing cosine similarity.

### Data Handling
- **JSON**: The knowledge base is stored in a JSON file (`kb1.json`) and loaded into the chatbot for retrieval.

### Web Scraping
- **BeautifulSoup**: Employed to scrape questions and answers from the MSEB website to enhance the knowledge base.

### Data Visualization
- **Matplotlib**: Used to visualize performance metrics through bar plots.

## Project Components

### Knowledge Base
- The knowledge base is stored in a JSON file (`kb1.json`) containing pairs of questions and corresponding answers.

### Text Preprocessing
- **Tokenization**: Breaking down text into individual words or tokens.
- **Lemmatization**: Reducing words to their base or root form.

### Vectorization and Similarity Measurement
- **TF-IDF Vectorizer**: Converts text data into numerical vectors.
- **Cosine Similarity**: Measures the similarity between user queries and knowledge base questions to identify the closest match.

### Evaluation Metrics
- **Accuracy**: The proportion of correctly identified responses.
- **Precision**: The ratio of relevant responses among the retrieved ones.
- **Recall**: The ratio of relevant responses that were successfully retrieved.
- **F1 Score**: The harmonic mean of precision and recall.

### Performance Comparison
- Comparison between NLTK and spaCy for text preprocessing and their impact on the chatbot's performance.

## System Architecture

1. **User Input**: The user enters a query.
2. **Text Preprocessing**: The query is preprocessed using tokenization and lemmatization.
3. **Vectorization**: The preprocessed query is transformed into a numerical vector using TF-IDF.
4. **Similarity Calculation**: Cosine similarity is computed between the query vector and knowledge base vectors.
5. **Response Generation**: The chatbot identifies the most similar question in the knowledge base and returns the corresponding answer.
6. **Web Scraping**: BeautifulSoup is used to scrape additional questions and answers from the MSEB website to enrich the knowledge base.

## Future Scope

1. **Enhanced NLP Techniques**: Incorporate advanced NLP models like BERT or GPT for better query understanding and response generation.
2. **Voice Interface**: Implement a voice interface to make the chatbot more accessible.
3. **Real-Time Learning**: Enable the chatbot to learn from new queries and responses in real-time.
4. **Integration With Actual MSEB Website**: Seamlessly integrate the chatbot with the MSEB website for direct user interaction.
5. **Multi-Language Support**: Extend the chatbot to support multiple languages, catering to a wider audience.

## Conclusion

The MSEB Chatbot project demonstrates the practical application of natural language processing and machine learning techniques to build an effective information-seeking agent. Through the integration of a comprehensive knowledge base and sophisticated text processing algorithms, the chatbot provides accurate and timely responses, enhancing user experience and operational efficiency.

## How to Run the Project

1. **Install Dependencies**: Ensure you have Python and the necessary libraries installed.
   ```sh
   pip install spacy nltk scikit-learn matplotlib beautifulsoup4
2. **Download NLTK Data**: Download the required NLTK datasets.
   ```sh
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')p.
