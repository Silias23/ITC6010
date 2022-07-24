import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from amazon_sentiment_analysis import stopwords_processing


def multinomial_nb_count_vectorizer(x_train, y_train, x_test, y_test, stop_words):
    
    model = make_pipeline(CountVectorizer(ngram_range=(1, 3), stop_words=stop_words), MultinomialNB())
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print("Model: CountVectorizer + MultinomialNB w/o stopwords")
    print(f"Number of features: {len(model[0].vocabulary_)}")
    print("Accuracy: {:.4f}\n".format(accuracy_score(y_test, preds)))
    mat = confusion_matrix(y_test, preds)
    ax = plt.axes()
    sns.heatmap(mat, square=True, annot=True, fmt='d', xticklabels=model.classes_,
                yticklabels=model.classes_,ax = ax)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    ax.set_title('CountVectorizer + MultinomialNB w/o stopwords')
    plt.show()

    print(classification_report(y_test, preds, labels=model.classes_))


def multinomial_nb_tfidf_vectorizer(x_train, y_train, x_test, y_test, stop_words):
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 3), stop_words=stop_words), MultinomialNB())

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print("Model: TfidfVectorizer + MultinomialNB w/o stopwords")
    print(f"Number of features: {len(model[0].vocabulary_)}")
    print("Accuracy: {:.4f}\n".format(accuracy_score(y_test, preds)))
    mat = confusion_matrix(y_test, preds)
    ax = plt.axes()
    sns.heatmap(mat, square=True, annot=True, fmt='d', xticklabels=model.classes_,
                yticklabels=model.classes_, ax=ax)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    ax.set_title('TfidfVectorizer + MultinomialNB w/o stopwords')
    plt.show()
    print(classification_report(y_test, preds, labels=model.classes_))


if __name__ == "__main__":
    df = pd.read_csv('data/text_title.csv')
    train_len = 50000
    #test_len = 10000

    stop_words = stopwords_processing()

    x_train, y_train = df['text'][:train_len], df['label'][:train_len].values
    x_test, y_test = df['text'][train_len:], df['label'][train_len:].values

    multinomial_nb_count_vectorizer(x_train, y_train, x_test, y_test, stop_words)
    multinomial_nb_tfidf_vectorizer(x_train, y_train, x_test, y_test, stop_words)

