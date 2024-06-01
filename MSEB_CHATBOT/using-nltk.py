import json
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load data from JSON file
with open("D:\BTech\TY 6th Sem\Artificial Intelligence\MSEB_CHATBOT\MSEB_CHATBOT\kb1.json", "r") as file:
    kb_data = json.load(file)

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    return nltk.word_tokenize(text.lower())

def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Preprocess the knowledge base data
processed_questions = [" ".join(lemmatize(tokenize(pair["question"]))) for pair in kb_data]
answers = [pair["answer"] for pair in kb_data]

# Vectorize the preprocessed questions
vectorizer = TfidfVectorizer(tokenizer=lemmatize, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# Generating response
def response(user_response):
    user_tfidf = vectorizer.transform([user_response])  # Pass user response directly for preprocessing

    # Calculate cosine similarities between user input and knowledge base
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)

    # Find the index of the most similar question
    idx = similarities.argmax()

    # Return the corresponding answer if similarity is above a threshold
    if similarities[0, idx] > 0.6:  # Adjust the threshold as needed
        return answers[idx]
    else:
        return "I am sorry! I don't have an answer to that question."

# Chatbot interaction loop
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while True:
    user_response = input().lower()
    if user_response == 'bye':
        print("ROBO: Bye! Take care..")
        break
    else:
        print("ROBO:", response(user_response))
