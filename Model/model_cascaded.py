import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from Model.preprocess import preprocess
import pickle
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import regularizers, optimizers, initializers

inputs, labels = preprocess()

# split training set into train set and test set (for purpose of quicker observation)
cv_size = 0.3
X_train, X_cv, y_train, y_cv = train_test_split(inputs, labels, test_size=cv_size)
# Encode labels
num_classes = 9
y_train_enc = label_binarize(y_train, classes=range(0, 9))
y_cv_enc = label_binarize(y_cv, classes=range(0, 9))

# Parameters
learning_rate = 0.0001
batch_size = 100
epochs = 100
drop_prob = 0.5
beta = 0.01
epsilon = 0.001

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons

# ---------------------------------- Stage One --------------------------------------
model_path_xgb = 'D:/kaggle-otto/Model/otto_xgb.pickle.dat'
model_path_ann = 'D:/kaggle-otto/Model/model_data/model_ann.ckpt'

num_input_ann = X_train.shape[1]  # number of features

tf.reset_default_graph()

# tf Graph input
X = tf.placeholder("float", [None, num_input_ann])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.get_variable("W_1", shape=[num_input_ann, n_hidden_1]),
    'h2': tf.get_variable("W_2", shape=[n_hidden_1, n_hidden_2]),
    'out': tf.get_variable("W_OUT", shape=[n_hidden_2, num_classes])
}
biases = {
    'b1': tf.get_variable("B_1", shape=[n_hidden_1]),
    'b2': tf.get_variable("B_2", shape=[n_hidden_2]),
    'out': tf.get_variable("B_OUT", shape=[num_classes])
}


def model_ann(x):
    # Hidden fully connected layer with 512 neurons, Relu activation
    z1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    batch_mean_1, batch_var_1 = tf.nn.moments(z1, [0])
    scale_bn_1 = tf.Variable(tf.ones([n_hidden_1]))
    beta_bn_1 = tf.Variable(tf.zeros([n_hidden_1]))
    bn1 = tf.nn.batch_normalization(z1, batch_mean_1, batch_var_1, beta_bn_1, scale_bn_1, epsilon)
    layer_1 = tf.nn.relu(bn1)
    # Hidden fully connected layer with 256 neurons, Relu activation
    z2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    batch_mean_2, batch_var_2 = tf.nn.moments(z2, [0])
    scale_bn_2 = tf.Variable(tf.ones([n_hidden_2]))
    beta_bn_2 = tf.Variable(tf.zeros([n_hidden_2]))
    bn2 = tf.nn.batch_normalization(z2, batch_mean_2, batch_var_2, beta_bn_2, scale_bn_2, epsilon)
    layer_2 = tf.nn.relu(bn2)
    # Output fully connected layer with 1 neuron for each class
    z3 = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    batch_mean_3, batch_var_3 = tf.nn.moments(z3, [0])
    scale_bn_3 = tf.Variable(tf.ones([num_classes]))
    beta_bn_3 = tf.Variable(tf.zeros([num_classes]))
    out_layer = tf.nn.batch_normalization(z3, batch_mean_3, batch_var_3, beta_bn_3, scale_bn_3, epsilon)
    return out_layer


# Construct model
prediction = model_ann(X)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Predict with model_ann and model_xgb
with tf.Session() as sess:
    # Restore xgb and ann models
    xgb = pickle.load(open("D:/kaggle-otto/Model/otto_xgb.pickle.dat", "rb"))
    saver.restore(sess, model_path_ann)

    # Predict with restored models
    y_train_xgb = xgb.predict_proba(X_train)
    y_train_ann = sess.run(tf.nn.softmax(prediction), feed_dict={X: X_train})
    y_cv_xgb = xgb.predict_proba(X_cv)
    y_cv_ann = sess.run(tf.nn.softmax(prediction), feed_dict={X: X_cv})
    X_train = np.concatenate((y_train_xgb, y_train_ann), axis=1)
    X_cv = np.concatenate((y_cv_xgb, y_cv_ann), axis=1)

# ---------------------------------- Stage Two --------------------------------------
input_dim = X_train.shape[1]

# Create model
model = Sequential()
# Layer 1
model.add(Dense(n_hidden_1,
                input_dim=input_dim,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                kernel_regularizer=regularizers.l2(beta),
                activity_regularizer=regularizers.l2(beta)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_prob))
# Layer 2
model.add(Dense(n_hidden_2,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                kernel_regularizer=regularizers.l2(beta),
                activity_regularizer=regularizers.l2(beta)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_prob))
# Output layer
model.add(Dense(num_classes,
                kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('softmax'))
# Compile model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=epsilon),
              metrics=['accuracy'])
# Train compiled model
model.fit(X_train, y_train_enc, epochs=100, batch_size=100)
# Print results
score_train = model.evaluate(X_train, y_train_enc, verbose=0)
score_cv = model.evaluate(X_cv, y_cv_enc, verbose=0)
print("\n------------- Model Report -------------")
print('Train score: {:.6f}'.format(score_train[0]))
print('Train accuracy: {:.6f}'.format(score_train[1]))
print('Validation score: {:.6f}'.format(score_cv[0]))
print('Validation accuracy: {:.6f}'.format(score_cv[1]))
# Save trained model
model.save('model_cascaded.h5')
