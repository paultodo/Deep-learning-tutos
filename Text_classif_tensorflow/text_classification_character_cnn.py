#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This is an example of using convolutional networks over characters
for DBpedia dataset to predict class from description of an entity.
This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626
and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
from tensorflow.contrib import learn

### Training data

# # Downloads, unpacks and reads DBpedia dataset.
# dbpedia = learn.datasets.load_dataset('dbpedia')
# X_train, y_train = pandas.DataFrame(dbpedia.train.data)[1], pandas.Series(dbpedia.train.target)
# X_test, y_test = pandas.DataFrame(dbpedia.test.data)[1], pandas.Series(dbpedia.test.target)


## LOAD IT LIKE YOU WOULD A REGULAR CSV FILE.
def load_data(sa = False) :
    if sa :
        print ('sentiment analysis')
        df_data = pandas.read_csv('/home/paul/TELECOM/NLP/tensorflow/data_tf/SentimentAnalysisDataset.csv', error_bad_lines = False)
        df_train = df_data[::2]
        df_test = df_data[1::2]
        X_train, y_train = df_train['SentimentText'], df_train['Sentiment']
        X_test, y_test = df_test['SentimentText'], df_test['Sentiment']
        
    else :
        print ('dbpedia')
        train = pandas.read_csv('/home/paul/TELECOM/NLP/tensorflow/data_tf/dbpedia_csv/train.csv', header=None)[:100000]
        X_train, y_train = train[2], train[0]
        test = pandas.read_csv('/home/paul/TELECOM/NLP/tensorflow/data_tf/dbpedia_csv/test.csv', header=None)[:100000]
        X_test, y_test = test[2], test[0]


    return X_test, y_test, X_train, y_train

X_test, y_test, X_train, y_train = load_data(sa = True)
### Process vocabulary

MAX_DOCUMENT_LENGTH = 100

char_processor = learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(char_processor.fit_transform(X_train)))
X_test = np.array(list(char_processor.transform(X_test)))

### Models

N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

def char_cnn_model(X, y):
    """Character level convolutional neural network model to predict classes."""
    byte_list = tf.reshape(learn.ops.one_hot_matrix(X, 256), 
        [-1, MAX_DOCUMENT_LENGTH, 256, 1])
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = learn.ops.conv2d(byte_list, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # Add a RELU for non linearity.
        conv1 = tf.nn.relu(conv1)
        # Max pooling across output of Convlution+Relu.
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1], 
            strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = learn.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    # Apply regular WX + B and classification.
    return learn.models.logistic_regression(pool2, y)

classifier = learn.TensorFlowEstimator(model_fn=char_cnn_model, n_classes=2,
    steps=100, optimizer='Adam', learning_rate=0.01, continue_training=True)

# Continuously train for 1000 steps & predict on test set.
score = 0
while score < 0.8 :
    classifier.fit(X_train, y_train)
    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print("Accuracy: %f" % score)