import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


batch_size = 16
df = pd.read_csv('data/text_title.csv')
train_len = 10000
# test_len = 10000
texts = df['text'].values.tolist()

labels = df['label'].values.tolist()
new_labels = [item - 1 for item in labels]
new_labels = np.array(new_labels)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(texts, new_labels, train_size=train_len,
                                                                              random_state=280493, stratify=new_labels)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Tokenizer

inputs_train = tokenizer(train_sentences, padding=True, truncation=True, return_tensors='tf')  # Tokenized text
inputs_test = tokenizer(test_sentences, padding=True, truncation=True, return_tensors='tf')  # Tokenized text

dataset_train = tf.data.Dataset.from_tensor_slices((dict(inputs_train), train_labels))  # Create a tensorflow dataset
dataset_test = tf.data.Dataset.from_tensor_slices((dict(inputs_test), test_labels))  # Create a tensorflow dataset
train_ds = dataset_train.batch(batch_size)
val_ds = dataset_test.batch(batch_size)

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=1)
preds = model.predict(val_ds)

tf_predictions = tf.nn.softmax(preds[0], axis=-1)
labels_ = ['class 0', 'class 1']
pred_labels = tf.argmax(tf_predictions, axis=1)
pred_labels = pred_labels.numpy()
pred_labels = pred_labels.tolist()

print("Accuracy: {:.4f}\n".format(accuracy_score(test_labels, pred_labels)))
mat = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sns.heatmap(mat, square=True, annot=True, fmt='d', xticklabels=[1, 2],
            yticklabels=[1, 2], ax=ax)
plt.xlabel('Predicted label')
plt.ylabel('True label')
ax.set_title('BERT')
plt.show()
print(classification_report(test_labels, pred_labels, target_names=labels_))