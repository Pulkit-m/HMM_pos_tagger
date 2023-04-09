import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class HMM_symbolic():
    def __init__(self):
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
    
    def process(self,word):
            word = word.lower()
            #remove punctuations if word is not a special character
            if word not in ['.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "'", '"', '``', "''", '--', '...']:
                word = word.strip(".,?;:()[]{}!-")
                #remove possessive 's
                if word.endswith("'s") and len(word) > 2:
                    word = word[:-2]
                #remove ' at the end of the word
                word = word.strip("'")
            return word
    
    def preprocess(self):
        #for all the sentences in the self.tagged_sentences, lowercase the words
        print("Preprocessing the data...")
        self.sentences = [[self.process(word) for word in sentence] for sentence in self.sentences]

    def create_bigrams(self):
        #create bigrams of the tags and words from scratch
        print("Creating bigrams of the words and tags...")
        tag_bigrams = []
        word_tag_bigrams = []
        for sentence,tag_list in zip(self.sentences,self.tags):
            for i in range(len(sentence)-1):
                tag_bigrams.append((tag_list[i],tag_list[i+1]))
                word_tag_bigrams.append((sentence[i],tag_list[i]))
            word_tag_bigrams.append((sentence[-1],tag_list[-1]))
        return tag_bigrams, word_tag_bigrams

    def fit(self,sentences,tags):
        print("Fitting the model...\n")
        #setting the instance variables
        self.sentences = sentences
        self.tags = tags
        #preprocess the data
        self.preprocess()
        self.tag_set = set([tag for _ in self.tags for tag in _])
        self.vocab = set([word for _ in self.sentences for word in _])
        self.tag2idx = {tag:idx for idx,tag in enumerate(self.tag_set)}
        self.idx2tag = {idx:tag for idx,tag in enumerate(self.tag_set)}
        self.vocab2idx = {word:idx for idx,word in enumerate(self.vocab)}
        self.idx2vocab = {idx:word for idx,word in enumerate(self.vocab)}
        self.N = len(self.tag_set)
        self.V = len(self.vocab)
        self.initial_matrix = np.zeros(self.N, dtype='float64')
        self.transition_matrix = np.zeros((self.N,self.N), dtype='float64')
        self.emission_matrix = np.zeros((self.N,self.V), dtype='float64')

        print("Populating the initial matrix...")
        for tag_list in self.tags:
            self.initial_matrix[self.tag2idx[tag_list[0]]] += 1
        
        #create bigrams of tags and words
        tag_bigrams, word_tag_bigrams = self.create_bigrams()
        #populate transition matrix
        print("Populating the transition matrix...")
        for prev_tag,next_tag in tag_bigrams:
            self.transition_matrix[self.tag2idx[prev_tag]][self.tag2idx[next_tag]] += 1
        #populate emission matrix
        print("Populating the emission matrix...")
        for word,tag in word_tag_bigrams:
            tag_idx = self.tag2idx[tag]
            word_idx = self.vocab2idx[word]
            self.emission_matrix[tag_idx][word_idx] += 1

        self.initial_matrix = self.initial_matrix / self.initial_matrix.sum() 
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        self.emission_matrix = self.emission_matrix / self.emission_matrix.sum(axis=1, keepdims=True)
    
    def emission(self, word, tag):
        if word in self.vocab:
            return self.emission_matrix[self.tag2idx[tag]][self.vocab2idx[word]]
        else:
            #return the probability of the most probable tag for the word
            return np.max(self.emission_matrix[self.tag2idx[tag],:])

    def Viterbi(self, sentence):
        #initialization
        T = len(sentence)
        N = self.N
        viterbi = np.zeros((T, N), dtype='float64')
        backpointer = np.zeros((T, N), dtype='int')
        #initialization step
        for s in range(N):
            viterbi[0][s] = self.initial_matrix[s] * self.emission(sentence[0], self.idx2tag[s])
            backpointer[0][s] = 0
        #recursion step
        for t in range(1, T):
            for s in range(N):
                viterbi[t][s] = np.max(viterbi[t-1] * self.transition_matrix[:,s]) * self.emission(sentence[t], self.idx2tag[s])
                backpointer[t][s] = np.argmax(viterbi[t-1] * self.transition_matrix[:,s])
        #backtracking step
        best_path_pointer = np.argmax(viterbi[T-1])
        best_path = [best_path_pointer]
        for t in range(T-1, 0, -1):
            best_path_pointer = backpointer[t, int(best_path_pointer)]
            best_path.append(best_path_pointer)
        best_path.reverse()
        return [self.idx2tag[idx] for idx in best_path]
    
    def predict(self, sentences):
        predictions = []
        print("Predicting...")
        for sentence in sentences:
            predictions.append(self.Viterbi(sentence))
        return predictions
    
    def evaluate(self, sentences, tags):
        sentences = [[self.process(word) for word in sentence] for sentence in sentences]
        predictions = self.predict(sentences)
        y_true = []
        y_pred = []
        for i in range(len(tags)):
            for j in range(len(tags[i])):
                y_true.append(tags[i][j])
                y_pred.append(predictions[i][j])

        total_accuracy = accuracy_score(y_true, y_pred)
        results = {"accuracy": total_accuracy}
        confusion_mat = confusion_matrix(y_true, y_pred, labels=list(self.tag_set))
        results["confusion_matrix"] = confusion_mat
        results["tag_set"] = list(self.tag_set)
        return results
    
    def training_results(self):
        return self.evaluate(self.sentences, self.tags)

def separate_sentence_from_tags(sentences_tagged):
        #return sentence and tags separately
        sentences = []
        tags = []
        for sentence in sentences_tagged:
            new_sentence = []
            new_tags = []
            for word, tag in sentence:
                new_sentence.append(word)
                new_tags.append(tag)
            sentences.append(new_sentence)
            tags.append(new_tags)
        return sentences, tags
