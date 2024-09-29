import os
import tarfile
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import time

# 1. Download and extract the dataset
url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
file_name = 'rt-polaritydata.tar.gz'
data_dir = './rt-polaritydata/'

# Download the data if it doesn't exist
if not os.path.exists(file_name):
    r = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(r.content)

# Extract the data if it doesn't exist
if not os.path.exists(data_dir):
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall()

# File paths for positive and negative reviews
pos_file_path = os.path.join(data_dir, 'rt-polarity.pos')
neg_file_path = os.path.join(data_dir, 'rt-polarity.neg')

# 2. Read the data
def read_data(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        return [line.strip() for line in file.readlines()]

positive_sentences = read_data(pos_file_path)
negative_sentences = read_data(neg_file_path)

# 3. Data Cleaning
positive_sentences = [sentence.lower() for sentence in positive_sentences]
negative_sentences = [sentence.lower() for sentence in negative_sentences]
positive_sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in positive_sentences]
negative_sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in negative_sentences]

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

positive_sentences = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in positive_sentences]
negative_sentences = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in negative_sentences]

stemmer = PorterStemmer()
positive_sentences = [' '.join([stemmer.stem(word) for word in sentence.split()]) for sentence in positive_sentences]
negative_sentences = [' '.join([stemmer.stem(word) for word in sentence.split()]) for sentence in negative_sentences]

# 4. Create train, validation, and test sets
pos_train, pos_temp = train_test_split(positive_sentences, train_size=4000, shuffle=False)
pos_val, pos_test = train_test_split(pos_temp, train_size=500, shuffle=False)
neg_train, neg_temp = train_test_split(negative_sentences, train_size=4000, shuffle=False)
neg_val, neg_test = train_test_split(neg_temp, train_size=500, shuffle=False)

# Combine data and labels
train_texts = pos_train + neg_train
train_labels = [1] * 4000 + [0] * 4000
val_texts = pos_val + neg_val
val_labels = [1] * 500 + [0] * 500
test_texts = pos_test + neg_test
test_labels = [1] * 831 + [0] * 831

# 5. Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

# 6. Define and train multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

print("Training models...")
time.sleep(1)  # Simulate loading time
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, train_labels)

    # 7. Evaluate on validation set
    val_predictions = model.predict(X_val)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='binary')
    
    # Append the results
    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Create a DataFrame to display results
results_df = pd.DataFrame(results)

# 8. Display results
print("\n### Model Performance ###")
print(results_df)

# 9. Evaluate the best model on the test set
best_model_name = results_df.loc[results_df['F1 Score'].idxmax()]['Model']
best_model = models[best_model_name]

test_predictions = best_model.predict(X_test)

# Classification report for the test set
print(f"\n### Best Model: {best_model_name} ###")
print("#### Test Set Report ####")
report = classification_report(test_labels, test_predictions, output_dict=True)
print(pd.DataFrame(report).T)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()
print(f"\n#### Confusion Matrix for {best_model_name} ####")
print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")

# Precision, recall, and F1-score for the test set
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average='binary')
print(f"#### Test Set Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
