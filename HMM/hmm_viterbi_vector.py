import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class HMM_vector():
    def __init__(self) -> None:
        """
        Initialize the HMM model
        self.sentences: list of lists of sentences
        self.tags: list of lists of tags corresponding to the sentences
        self.tag_set: set of unique tags in the corpus
        self.vocab: list of unique words in the corpus
        self.tag2idx: dictionary of tags to indices
        self.idx2tag: dictionary of indices to tags
        self.vocab2idx: dictionary of words to indices
        self.idx2vocab: dictionary of indices to words
        self.N: number of tags
        self.V: size of vocabulary
        self.initial_matrix: initial probability matrix
        self.transition_matrix: transition probability matrix
        self.emission_matrix: emission probability matrix
        """
        self.sentences = []
        self.tag_set = set()
        self.tags = []
        self.vocab = []
        self.tag2idx = {}
        self.idx2tag = {}
        self.vocab2idx = {}
        self.idx2vocab = {}
        self.N = 0
        self.V = 0
        self.initial_matrix = None
        self.transition_matrix = None
        self.emission_matrix = None