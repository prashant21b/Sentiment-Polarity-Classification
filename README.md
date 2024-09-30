
# Sentiment Polarity Classification

This project implements a binary sentiment polarity classifier using the Cornell Movie Review dataset. The classifier is trained to differentiate between positive and negative movie reviews.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction
This project implements multiple machine learning models to classify movie reviews as either positive or negative. The dataset contains an equal number of positive and negative sentences. The models used in this project include:
- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier (SVC)
- Gradient Boosting Classifier

## Dataset
The dataset is provided by Cornell University and contains 5,331 positive and 5,331 negative movie reviews.

- **Download Link**: [Movie Review Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)

The dataset is split into training, validation, and test sets as follows:
- **Training Set**: First 4,000 positive and 4,000 negative sentences
- **Validation Set**: Next 500 positive and 500 negative sentences
- **Test Set**: Final 831 positive and 831 negative sentences

## Preprocessing
Before training the models, several preprocessing steps are performed:
1. **Lowercasing**: All sentences are converted to lowercase.
2. **Punctuation Removal**: Non-alphanumeric characters are removed using regular expressions.
3. **Stopword Removal**: Common English stopwords are filtered out using NLTK's `stopwords` corpus.
4. **Stemming**: Words are reduced to their base form using the Porter Stemmer.

## Models
The following models are implemented and trained:
- **Logistic Regression**: A linear model for binary classification.
- **Random Forest Classifier**: An ensemble model that uses multiple decision trees.
- **Support Vector Classifier (SVC)**: A classifier that finds the hyperplane separating the classes.
- **Gradient Boosting Classifier**: A model that builds successive decision trees, minimizing classification error.

Each model is trained using the TF-IDF (Term Frequency-Inverse Document Frequency) representation of the text data, with a maximum of 5,000 features.

## Evaluation
Model performance is evaluated on the validation and test sets using the following metrics:
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix** (True Positives, False Positives, True Negatives, False Negatives)

The best-performing model on the validation set is selected and evaluated on the test set.

## Installation
### Dependencies
To run the code, you will need Python 3.x and the following libraries:
- `requests`
- `nltk`
- `sklearn`
- `pandas`
- `numpy`

To install the required packages, run:
```bash
pip install -r requirements.txt
```

### NLTK Stopwords
Ensure that NLTK stopwords are downloaded by running the following commands:
```python
import nltk
nltk.download('stopwords')
```

## Usage
1. Download the dataset from the provided link.
2. Run the script to train models and evaluate their performance:
   ```bash
   python sentiment_review.py
   ```

The script will output performance metrics for each model, and the best model will be evaluated on the test set.

## Results
The results of the models are summarized below:

| Model                 | Precision | Recall | F1 Score |
|-----------------------|-----------|--------|----------|
| Logistic Regression    | 0.7808    | 0.7480  | 0.7640   |
| Random Forest          | 0.7372    | 0.6620  | 0.6976   |
| Support Vector Classifier (SVC) | 0.7871 | 0.7540 | 0.7702 |
| Gradient Boosting Classifier | 0.6176 | 0.7880 | 0.6924  |

The confusion matrix for the best model is as follows:
- True Positives: 613
- True Negatives: 645
- False Positives: 186
- False Negatives: 218

## References
- [Movie Review Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)
- [Cornell Movie Review Data README](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.README.1.0.txt)
