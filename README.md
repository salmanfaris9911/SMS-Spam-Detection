# Spam Detection using SVM
This project aims to classify SMS messages as spam or not spam using a Support Vector Machine (SVM) model. The dataset used is the SMS Spam Collection dataset from Kaggle, and the implementation is done in Python using libraries like pandas, scikit-learn, and NLTK.

# Dataset
The dataset is the SMS Spam Collection Dataset from Kaggle. It contains 5,572 SMS messages labeled as either "spam" or "ham" (not spam). The data is stored in a CSV file (spam.csv) with columns v1 (label) and v2 (message text), along with some unnamed columns that are dropped during preprocessing.

# Data Preprocessing
The following steps were applied to clean and prepare the data for modeling:

Dropped unnecessary columns: Removed 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4' from the dataset.
Label encoding: Encoded the target variable v1 (spam/ham) into numerical values (0 for ham, 1 for spam) using LabelEncoder.
Removed duplicates: Kept only the first occurrence of duplicate messages.
Text preprocessing:
Removed punctuation from the messages.
Converted all text to lowercase.
Tokenized the text into individual words.
Removed stopwords using NLTK's English stopwords list.
Applied stemming with PorterStemmer.
Removed numbers from the text.
Applied lemmatization using WordNetLemmatizer.
Feature extraction: Used TfidfVectorizer to convert the preprocessed text into numerical features with the following settings:
max_features=1500
min_df=5
max_df=0.7

# Model Training and Evaluation
Data split: The dataset was split into training and testing sets with an 80/20 split (test_size=0.2, random_state=0).
Model: Trained a Support Vector Machine (SVM) classifier with an RBF kernel (kernel='rbf') using scikit-learn's SVC.
Evaluation metrics: The model's performance was assessed using accuracy, confusion matrix, and classification report on the test set.
Results
The model achieved the following performance on the test set:

Accuracy: 0.9768
Confusion Matrix:
[[883   2]
[ 22 127]]
True Negatives (ham correctly classified): 883
False Positives (ham misclassified as spam): 2
False Negatives (spam misclassified as ham): 22
True Positives (spam correctly classified): 127
Classification Report:
precision    recall  f1-score   support

       0       0.98      1.00      0.99       885
       1       0.98      0.85      0.91       149

accuracy                           0.98      1034
macro avg       0.98      0.93      0.95      1034
weighted avg       0.98      0.98      0.98      1034
