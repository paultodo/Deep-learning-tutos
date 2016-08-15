import pandas
import numpy as np
import tensorflow as tf
from tensorflow.contrib import skflow
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import random
from sklearn.preprocessing import StandardScaler
from pprint import pprint
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()



# ### DNN ###

random.seed(42)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3, batch_size=128, steps = 500, learning_rate=0.05, dropout = 0.5)
classifier.fit(iris.data,iris.target, logdir ='/tmp/tf_examples/my_model_1/')
score = accuracy_score(classifier.predict(iris.data),iris.target)
print "DNN"
print('Accuracy: {0:f}'.format(score))



# ### CUSTOM DNN MODEL ###

def my_model(X, y):
    """This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability."""
    layers = skflow.ops.dnn(X, [10, 20, 10], dropout=0.5)
    return skflow.models.logistic_regression(layers, y)

classifier = skflow.TensorFlowEstimator(model_fn=my_model, n_classes=3)
classifier.fit(iris.data, iris.target)
score = accuracy_score(iris.target, classifier.predict(iris.data))
print "Custom DNN"
print('Accuracy: {0:f}'.format(score))


### RNN ###

train = pandas.read_csv('/home/paul/TELECOM/NLP/tensorflow/data_tf/dbpedia_csv/train.csv', header=None)[:1000]
X_train, y_train = train[2], train[0]
test = pandas.read_csv('/home/paul/TELECOM/NLP/tensorflow/data_tf/dbpedia_csv/test.csv', header=None)[:1000]
X_test, y_test = test[2], test[0]

MAX_DOCUMENT_LENGTH = 10

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

n_words = len(vocab_processor.vocabulary_)


print('Total words: %d' % n_words)

EMBEDDING_SIZE = 50

def my_input_op_fn(X):
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
    return word_list


classifier = skflow.TensorFlowRNNClassifier(rnn_size=EMBEDDING_SIZE, 
    n_classes=15, cell_type='gru', input_op_fn=my_input_op_fn,
    num_layers=2, bidirectional=False, sequence_length=None,
    steps=200, optimizer='Adam', learning_rate=0.01, continue_training=True)
classifier.fit(X_train,y_train, logdir ='/tmp/tf_examples/my_model_1/')
score = accuracy_score(classifier.predict(X_test),y_test)
print "RNN"
print('Accuracy: {0:f}'.format(score))


## DNN with pipeline ###

# It's useful to scale to ensure Stochastic Gradient Descent will do the right thing
scaler = StandardScaler()

# DNN classifier
DNNclassifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3, steps=200)

pipeline = Pipeline([('scaler', scaler), ('DNNclassifier', DNNclassifier)])
pipeline.fit(iris.data, iris.target)
score = accuracy_score(iris.target, pipeline.predict(iris.data))

print('Accuracy: {0:f}'.format(score))


