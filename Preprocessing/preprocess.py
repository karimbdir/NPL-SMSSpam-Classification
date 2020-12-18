
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


def parse_text_file(path):
    df = pd.read_csv(path, delimiter='\t', names=['class', 'message'])
    return df

def describe(df):
    return df.describe()

def explore_class(df):
    return df.groupby('class').describe()

def messages_length(df):
    df['length'] = df['message'].apply(len)
    return df

def missing(df):
    return df.info()

def length_describe(df):
    return df['length'].describe()

def longest_message(df):
    return df[df['length'] == 910 ]['message'].iloc[0]

def count_class(df):
    class_count = pd.value_counts(df['class'],sort=True)
    class_count.plot(kind='bar', color = ['blue','green'])
    plt.title('Class Count')
    plt.show()

def count_class_hist(df):
    df.hist(column='length',by='class',bins=100,figsize=(10,5))
    plt.show()

def pie_class(df):
    class_count = pd.value_counts(df['class'], sort=True)
    class_count.plot(kind='pie', autopct='%2.0f%%')
    plt.show()

def text_process(text):
    text_process = text.lower()
    text_process = re.sub(r'\W+|\d+|_', ' ',text_process) #to remove numbers and punctuations
    tokens = word_tokenize(text_process)
    stopword = stopwords.words('english')
    filtered_sentence = [word for word in tokens if word not in stopword]
    lem = WordNetLemmatizer()
    lemma_tokens = [lem.lemmatize(word) for word in filtered_sentence]
    return text_process

def vectorization(text_process):
    vector_fitted = CountVectorizer(analyzer=text_process)
    return vector_fitted

def tfidf_function():
    tfidf_fitted = TfidfTransformer()
    return tfidf_fitted

def split_data(X,y):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    return X_train,X_test,y_train,y_test

def NB(X_train,X_test,y_train,y_test):
    model = naive_bayes.MultinomialNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('Naive Bayes ')
    print('Classification report' , classification_report(y_test,y_pred))
    print('Accuracy score', accuracy_score(y_pred,y_test))

def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('Logistic Regression')
    print('Classification report', classification_report(y_test, y_pred))
    print('Accuracy score', accuracy_score(y_pred, y_test))

def support_vector_machine(X_train, X_test, y_train, y_test):
    model = svm.SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('SVM')
    print('Classification report', classification_report(y_test, y_pred))
    print('Accuracy score', accuracy_score(y_pred, y_test))

def ada(X_train, X_test, y_train, y_test):
    model = ensemble.AdaBoostClassifier
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('AdaBoost')
    print('Classification report', classification_report(y_test, y_pred))
    print('Accuracy score', accuracy_score(y_pred, y_test))

def forest_tree(X_train, X_test, y_train, y_test):
    model = ensemble.RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Random Forest')
    print('Classification report', classification_report(y_test, y_pred))
    print('Accuracy score', accuracy_score(y_pred, y_test))

def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('Decision Tree')
    print('Classification report' , classification_report(y_test,y_pred))
    print('Accuracy score', accuracy_score(y_pred,y_test))
