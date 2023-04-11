import nltk
import string
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

def train_embeddings():
    # Download the Brown corpus and stopwords
    nltk.download('brown')
    nltk.download('stopwords')

# Load the Brown corpus
    sentences = nltk.corpus.brown.sents()

# Preprocess the sentences
    stemmed_sentences = embed_preprocessing(sentences)

# Train the word embeddings using the Skip-Gram model
    model = Word2Vec(stemmed_sentences, sg=1, vector_size=100, window=5, min_count=5, workers=4)

    #check if current directory is HMM otherwise change directory to HMM
    if os.getcwd().split('\\')[-1] != 'HMM':
        os.chdir('HMM')
    model_path = os.join(os.getcwd(),'brown_skipgram_preprocessed.model')
# Save the trained embeddings to a file
    model.save(model_path)
    return model

def embed_preprocessing(sentences):
    lowercase_sentences = [[word.lower() for word in sentence] for sentence in sentences]
    no_punc_sentences = [[word for word in sentence if word not in string.punctuation] for sentence in lowercase_sentences]
    stop_words = set(stopwords.words('english'))
    no_stop_sentences = [[word for word in sentence if word not in stop_words] for sentence in no_punc_sentences]
    porter_stemmer = PorterStemmer()
    stemmed_sentences = [[porter_stemmer.stem(word) for word in sentence] for sentence in no_stop_sentences]
    return stemmed_sentences