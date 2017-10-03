from __future__ import print_function

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow as tf


path = '../'
fname_train = 'train.csv'

# load training set
train = pd.read_csv(path + fname_train)
# drop 'id' column from training set
train.drop('id', axis=1, inplace=True)
# extract data in the 'target' column and convert them to numeric
y_train = train['target'].values
le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train)
# drop 'target' column from training set
train.drop('target', axis=1, inplace=True)
X_train = train.values

# split training set into train set and test set (for purpose of quicker observation)
test_size = 0.1
train_data, test_data, train_predictions, test_predictions = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=test_size)

# Binarize predictions
y_b_train = label_binarize(train_predictions, classes=range(0, 9))
y_b_test = label_binarize(test_predictions, classes=range(0, 9))

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 93 # number of features
num_classes = 9 # number of classes

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
# loss_op = tf.losses.log_loss(labels=y_b_train, predictions=Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        if len(train_data) >= batch_size:
            batch_x = train_data[:batch_size]
            train_data = train_data[batch_size:]
            batch_y = y_b_train[:batch_size]
            y_b_train = y_b_train[batch_size:]
        else:
            batch_x = train_data
            batch_y = y_b_train

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for test set
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data,
                                      Y: y_b_test}))