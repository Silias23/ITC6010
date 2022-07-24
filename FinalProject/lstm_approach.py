import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import keras
import nltk
from amazon_sentiment_analysis import stopwords_processing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
'''nltk.download('wordnet')
nltk.download('omw-1.4')
'''


def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st


df = pd.read_csv('data/text_title.csv')
train_len = 50000
# test_len = 10000

"""Preprocess for LSTM"""
stop_words = stopwords_processing()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df['text'] = df.text.apply(lemmatize_text)


texts = df['text'].values
labels = df['label'].values
new_labels = [item - 1 for item in labels]
new_labels = np.array(new_labels)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(texts, new_labels, train_size=50000,
                                                                              random_state=280493, stratify=new_labels)
print(len(train_sentences))
print(len(test_sentences))

# Hyperparameters of the model
vocab_size = 3000  # choose based on statistics
oov_tok = ''
embedding_dim = 128
max_length = 200  # choose based on statistics, for example 150 to 200
padding_type = 'post'
trunc_type = 'post'
# tokenize sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
# convert train dataset to sequence and pad sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)
# convert Test dataset to sequence and pad sequences
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)

# model initialization
model = keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model summary
model.summary()

num_epochs = 15
history = model.fit(train_padded, train_labels,
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=(test_padded, test_labels))

preds = model.predict(test_padded)
pred_labels = []
for i in preds:
    if i >= 0.5:
        pred_labels.append(1)
    else:
        pred_labels.append(0)
print("Model: LSTM")
print("Accuracy: {:.4f}\n".format(accuracy_score(test_labels, pred_labels)))
mat = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sns.heatmap(mat, square=True, annot=True, fmt='d', xticklabels=[1, 2],
            yticklabels=[1, 2], ax=ax)
plt.xlabel('Predicted label')
plt.ylabel('True label')
ax.set_title('LSTM')
plt.show()
print(classification_report(test_labels, pred_labels, target_names=['1', '2']))
