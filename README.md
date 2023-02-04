# Artificial Intelligence Project

## Workgroup
- Nicola Deidda
- Luca Minnei


## Dataset
20 Newsgroups:
http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz


## Requirements
1) remove stop words from every document (i.e., generic words that have no discriminant capability in text categorisation tasks, like "and", "the", etc.): you can easily find some software tool on the web

2) apply stemming to every remaining word in every document (i.e., reducing every word to its root): I suggest Porter stemmer algorithm, here is a Python implementation:
https://pypi.org/project/PorterStemmer/

3) apply 5-fold cross-validation on the data set: randomly subdivide it into 5 disjoint subsets of identical size, and repeate the steps below for each of the 5 training folds (made up of four subsets) and corresponding testing folds (the remaining subset)

3.1) extract the set of all distinct words from the *training* fold, and compute the discriminant capability of each word using, for instance, the Information Gain measure defined in section 5.4.2 of the attached paper (which is very easy to implement); then select the first N words as features (you can try different values for N in distinct experiments, for comparison, e.g., N = 100, 500, 1000)

3.2) compute a N-dimensional feature vector for every document in the training and testing folds, e.g., as the word occurrence (1 if a word is present in the email body, 0 otherwise), frequency, or TF-IDF (see the attached paper): you can choose one kind of feature, or maybe use and compare them all

3.3) train a classifier on the training fold, and compute the number of misclassifications on the corresponding testing fold

4) finally, estimate the misclassification rate as the sum of the number of misclassified documents in the 5 testing folds, divided by the overall number of documents


- link https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
