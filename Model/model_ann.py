from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from pathlib import Path
import shutil

# Delete old model data and event summaries
dir1 = './model_data'
path1 = Path(dir1)
dir2 = 'C:/Users/Think/AnacondaProjects/tmp/otto'
path2 = Path(dir2)
if path1.is_dir():
    shutil.rmtree(dir1)
if path2.is_dir():
    shutil.rmtree(dir2)

path = '../'
fname_train = 'train.csv'

# load training set
data = pd.read_csv(path + fname_train)
# drop 'id' column from training set
data.drop('id', axis=1, inplace=True)
# extract data in the 'target' column and convert them to numeric
labels = data['target'].values
le_y = LabelEncoder()
labels = le_y.fit_transform(labels)
# drop 'target' column from training set
data.drop('target', axis=1, inplace=True)
inputs = data.values

# split training set into train set and test set (for purpose of quicker observation)
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(inputs,
                                                    labels,
                                                    test_size=test_size)
# Encode lables
num_classes = 9
y_train_enc = label_binarize(y_train, classes=range(0, 9))
y_test_enc = label_binarize(y_test, classes=range(0, 9))

# Parameters
learning_rate = 0.0001
batch_size = 100
epochs = 200
display_step = 1
keep_prob = 0.75
model_path = './model_data/model_ann.ckpt'

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 93  # number of features

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], mean=0.0, stddev=0.05, seed=123), name='W_1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0.0, stddev=0.05, seed=123), name='W_2'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]), name='W_OUT')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='B_1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='B_2'),
    'out': tf.Variable(tf.random_normal([num_classes]), name='B_OUT')
}

# Add histogram summaries for weights
tf.summary.histogram('w_h1_summ', weights['h1'])
tf.summary.histogram('w_h2_summ', weights['h2'])
tf.summary.histogram('w_out_summ', weights['out'])

# Add histogram summaries for biases
tf.summary.histogram('b_h1_summ', biases['b1'])
tf.summary.histogram('b_h2_summ', biases['b2'])
tf.summary.histogram('b_out_summ', biases['out'])


def neural_net_model(x):
    # Hidden fully connected layer with 512 neurons, Relu activation, 0.75 dropout
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), keep_prob=keep_prob)
    # Hidden fully connected layer with 256 neurons, Relu activation, 0.75 dropout
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), keep_prob=keep_prob)
    # Output fully connected layer with 1 neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
prediction = neural_net_model(X)

with tf.name_scope("cost"):
    # Define loss and optimizer
    cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Add scalar summary for cost tensor
    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", accuracy)

# Merge all the summaries and write them out to './logs/nn_logs'
merged = tf.summary.merge_all()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    train_writer = tf.summary.FileWriter("C:/Users/Think/AnacondaProjects/tmp/otto/logs/nn_logs")
    train_writer.add_graph(sess.graph)

    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(len(X_train) / batch_size)
        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(y_train_enc, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x,
                                                          Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Record train set summary per epoch step
        summary = sess.run(merged, feed_dict={X: X_train,
                                              Y: y_train_enc})
        train_writer.add_summary(summary, epoch)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    train_writer.close()
    # Save model
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
