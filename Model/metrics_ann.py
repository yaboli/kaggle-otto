import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import label_binarize

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
_, X_cv, _, y_cv = train_test_split(inputs,
                                    labels,
                                    test_size=test_size)
# Encode lables
num_classes = 9
y_cv_enc = label_binarize(y_cv, classes=range(0, 9))

# Parameters
model_path = 'D:/kaggle-otto/Model/model_data/model_ann.ckpt'

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 93  # number of features

tf.reset_default_graph()
# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.get_variable("W_1", shape=[num_input, n_hidden_1]),
    'h2': tf.get_variable("W_2", shape=[n_hidden_1, n_hidden_2]),
    'out': tf.get_variable("W_OUT", shape=[n_hidden_2, num_classes])
}
biases = {
    'b1': tf.get_variable("B_1", shape=[n_hidden_1]),
    'b2': tf.get_variable("B_2", shape=[n_hidden_2]),
    'out': tf.get_variable("B_OUT", shape=[num_classes])
}


def neural_net_model(x):
    # Hidden fully connected layer with 512 neurons, Relu activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons, Relu activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with 1 neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
prediction = neural_net_model(X)

# Define loss op
cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y)))

# Define accuracy op
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Add ops to restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, model_path)
    # Evaluate costs and accuracies
    print("\n------------- Model report -------------")
    print("Cost (validation):", cost.eval({X: X_cv, Y: y_cv_enc}))
    print("Accuracy (validation):", accuracy.eval({X: X_cv, Y: y_cv_enc}))
