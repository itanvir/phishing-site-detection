import numpy as np
from collections import Counter
from math import log

class CharTfidfVectorizer:
    def __init__(self):
        self.corpus_char_count = Counter()
        self.vocab = []
        self.idf = {}

    def fit(self, documents):
        self.corpus_char_count = Counter("".join(documents))
        self.vocab = list(self.corpus_char_count.keys())
        
        num_documents = len(documents)
        for char in self.vocab:
            char_in_docs = sum(1 for doc in documents if char in doc)
            self.idf[char] = log(num_documents / (1 + char_in_docs))

    def transform(self, documents):
        tfidf_vectors = []
        
        for doc in documents:
            document_char_count = Counter(doc)
            tfidf_vector = np.zeros(len(self.vocab))
            
            for char, char_count in document_char_count.items():
                if char in self.vocab:
                    tf = char_count / len(doc)
                    tfidf = tf * self.idf[char]
                    tfidf_vector[self.vocab.index(char)] = tfidf
            
            tfidf_vectors.append(tfidf_vector)
        
        return tfidf_vectors

# Example usage
if __name__ == "__main__":
    # Example documents
    documents = ["hello world", "goodbye", "hello"]
    
    # Create and fit the CharTfidfVectorizer
    char_tfidf_vectorizer = CharTfidfVectorizer()
    char_tfidf_vectorizer.fit(documents)
    
    # Transform the given documents into TF-IDF vectors
    transformed_vectors = char_tfidf_vectorizer.transform(documents)
    
    for i, vector in enumerate(transformed_vectors):
        print(f"Document {i + 1} TF-IDF Vector: {vector}")
