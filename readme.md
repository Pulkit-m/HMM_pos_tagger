
# HMM POS Tagger

This project contains python implementations of the **HMM-Viterbi_Symbolic** and **HMM-Viterbi-Vector algorithms**. The performances of these models were tested on the **NLTK Brown Corpus**.


## Authors

- [Shubham Awasthi (2019UCH0024)](https://www.github.com/shubhamawasthi0301)
- [Pulkit Mahajan (2019UCS0073)](https://www.github.com/)

## Usage

Install dependencies from the terminal
```terminal
pip install -r requirements.txt
```
- Run the **Results.ipynb** notebook.
- The notebook will automatically download the **NLTK Brown Corpus** and train the models. 

    *( 1. Make sure that you have an active internet connection.
    2. The training process may take a few minutes.)*

- Two files named **HMM_symbolic_results.pkl** and **HMM_vector_results.pkl** should now be present in the **HMM** directory. 
- Fter executing the notebook, you will be able to see:
    * The overall **Accuracy** of both the models
    * The **Accuracy** of both the models for each **POS Tag** in the corpus
    * The **Precision**, **Recall** and **F1-score** of both the models for each **POS Tag** in the corpus
    * The **Confusion Matrix** for both the models 