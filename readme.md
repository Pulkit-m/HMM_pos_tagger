
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
To get the cross-validation results, run the **cross_validation.py** file in the terminal of your system by using: 

```terminal
cd HMM
```
followed by either
```terminal
python cross_validation.py
```
or
```
python3 cross_validation.py
```
Two files named **HMM_symbolic_results.pkl** and **HMM_vector_results.pkl** should now be present in the **HMM**
directory.

Run the **Results.ipynb** notebook to see the accuracy as well as the heatmap of the confusion matrix for the each of the algorithms.