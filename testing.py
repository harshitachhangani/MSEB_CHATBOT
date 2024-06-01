import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load English language model for spaCy
nlp_spacy = spacy.load("en_core_web_sm")

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load data from JSON file
with open("kb1.json", "r") as file:
    kb_data = json.load(file)

# Tokenization and Lemmatization for NLTK
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    return nltk.word_tokenize(text.lower())

def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Preprocess the knowledge base data for NLTK
processed_questions_nltk = [" ".join(lemmatize(tokenize(pair["question"]))) for pair in kb_data]
answers_nltk = [pair["answer"] for pair in kb_data]

# Vectorize the preprocessed questions for NLTK
vectorizer_nltk = TfidfVectorizer(tokenizer=lemmatize, stop_words='english')
tfidf_matrix_nltk = vectorizer_nltk.fit_transform(processed_questions_nltk)

# Preprocess the knowledge base data for SpaCy
processed_questions_spacy = [" ".join([token.lemma_.lower() for token in nlp_spacy(pair["question"]) if not token.is_stop and not token.is_punct]) for pair in kb_data]
answers_spacy = [pair["answer"] for pair in kb_data]

# Vectorize the preprocessed questions for SpaCy
vectorizer_spacy = TfidfVectorizer(tokenizer=lambda text: [token.lemma_.lower() for token in nlp_spacy(text) if not token.is_stop and not token.is_punct], stop_words='english')
tfidf_matrix_spacy = vectorizer_spacy.fit_transform(processed_questions_spacy)

# Generating response for NLTK
def response_nltk(user_response):
    user_tfidf = vectorizer_nltk.transform([user_response])  # Pass user response directly for preprocessing

    # Calculate cosine similarities between user input and knowledge base
    similarities = cosine_similarity(user_tfidf, tfidf_matrix_nltk)

    # Find the index of the most similar question
    idx = similarities.argmax()

    return answers_nltk[idx], similarities.max()

# Generating response for SpaCy
def response_spacy(user_response):
    user_tfidf = vectorizer_spacy.transform([user_response])  # Pass user response directly for preprocessing

    # Calculate cosine similarities between user input and knowledge base
    similarities = cosine_similarity(user_tfidf, tfidf_matrix_spacy)

    # Find the index of the most similar question
    idx = similarities.argmax()

    return answers_spacy[idx], similarities.max()

# Evaluate NLTK and SpaCy models
actual_responses = []
nltk_responses = []
spacy_responses = []
similarity_scores_nltk = []
similarity_scores_spacy = []

for question_answer_pair in kb_data:
    actual_responses.append(question_answer_pair["answer"])
    nltk_response, nltk_similarity = response_nltk(question_answer_pair["question"])
    spacy_response, spacy_similarity = response_spacy(question_answer_pair["question"])
    nltk_responses.append(nltk_response)
    spacy_responses.append(spacy_response)
    similarity_scores_nltk.append(nltk_similarity)
    similarity_scores_spacy.append(spacy_similarity)

# Evaluate performance metrics
nltk_accuracy = accuracy_score(actual_responses, nltk_responses)
nltk_precision = precision_score(actual_responses, nltk_responses, average='weighted', zero_division=1)
nltk_recall = recall_score(actual_responses, nltk_responses, average='weighted', zero_division=1)
nltk_f1 = f1_score(actual_responses, nltk_responses, average='weighted', zero_division=1)

spacy_accuracy = accuracy_score(actual_responses, spacy_responses)
spacy_precision = precision_score(actual_responses, spacy_responses, average='weighted', zero_division=1)
spacy_recall = recall_score(actual_responses, spacy_responses, average='weighted', zero_division=1)
spacy_f1 = f1_score(actual_responses, spacy_responses, average='weighted', zero_division=1)

# Displaying similarity scores
print("Average similarity score for NLTK:", sum(similarity_scores_nltk) / len(similarity_scores_nltk))
print("Average similarity score for SpaCy:", sum(similarity_scores_spacy) / len(similarity_scores_spacy))

# Plotting performance metrics
labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
nltk_scores = [nltk_accuracy, nltk_precision, nltk_recall, nltk_f1]
spacy_scores = [spacy_accuracy, spacy_precision, spacy_recall, spacy_f1]

x = range(len(labels))

plt.figure(figsize=(10, 6))

plt.bar(x, nltk_scores, width=0.35, label='NLTK', color='b', align='center')
plt.bar(x, spacy_scores, width=0.35, label='SpaCy', color='r', align='edge')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Performance Metrics Comparison')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0.9, 1.0)  # Set y-axis limits

plt.show()
