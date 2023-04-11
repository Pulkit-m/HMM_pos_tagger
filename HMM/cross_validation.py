from hmm_viterbi_symbolic import *
from hmm_viterbi_vector import *
import multiprocessing
from sklearn.model_selection import KFold
import nltk
import pickle
import time

def run_fold(fold_idx, sentences, tags, indices,results_dict, model=HMM_symbolic):
    train_indices, test_indices = indices[fold_idx]
    train_sentences = [sentences[i] for i in train_indices]
    train_tags = [tags[i] for i in train_indices]
    test_sentences = [sentences[i] for i in test_indices]
    test_tags = [tags[i] for i in test_indices]

    hmm = model()
    hmm.fit(train_sentences, train_tags)
    results = hmm.evaluate(test_sentences, test_tags)

    results_dict[fold_idx] = results

def run_kfold(sentences, tags, k=5, model=HMM_symbolic):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = [(train_idx, test_idx) for train_idx, test_idx in kf.split(sentences)]

    manager = multiprocessing.Manager()
    results_dict = manager.dict()

    with multiprocessing.Pool(processes=k) as pool:
        for fold_idx in range(k):
            pool.apply_async(run_fold, args=(fold_idx, sentences, tags, indices, results_dict, model))

        pool.close()
        pool.join()

    final_results = {'tag_set': results_dict[0]['tag_set'], 'confusion_matrix':[], 'accuracy':[]}
    for i in range(k):
        for metric in results_dict[i]:
            if metric != 'tag_set':
                final_results[metric].append(results_dict[i][metric])
            
    for key in final_results:
        if key == 'accuracy':
            final_results[key] = sum(final_results[key])/len(final_results[key])
        elif key == 'confusion_matrix':
            #calculate the average confusion matrix
            final_results[key] = np.mean(final_results[key], axis=0)

    return final_results

if __name__ == '__main__':
    #using the nltk brown corpus and the universal tagset
    print("Loading the Corpus...")
    tagged_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')
    
    #separate the sentences and tags
    sentences, tags = separate_sentence_from_tags(tagged_sentences)
    k = 5
    # print(f"Running {k}-fold cross-validation for HMM_symbolic ...")
    # start = time.time()
    # final_results = run_kfold(sentences, tags, k=k, model=HMM_symbolic)
    # end = time.time()
    # print(f"Time taken for {k}-fold cross-validation : ", end-start)
    # # Save the final_results dictionary to a file
    # with open('HMM_symbolic_results.pkl', 'wb') as f:
    #     pickle.dump(final_results, f)
    
    # run the below code after HMM_vector has been implemented
    print(f"Running {k}-fold cross-validation for HMM vector...")
    start = time.time()
    final_results = run_kfold(sentences, tags, k=k, model=HMM_vector)
    end = time.time()
    print(f"Time taken for {k}-fold cross-validation : ", end-start)
    with open('HMM_vector_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
