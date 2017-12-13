import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from scipy import interp
from Model.preprocess import preprocess


inputs, labels = preprocess()

# split training set into train set and test set (for purpose of quicker observation)
test_size = 0.3
_, X_cv, _, y_cv = train_test_split(inputs, labels, test_size=test_size)
# Encode labels
num_classes = 9
y_cv_enc = label_binarize(y_cv, classes=range(0, 9))

# Parameters
model_path = 'D:/kaggle-otto/Model/model_data/model_ann.ckpt'

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = X_cv.shape[1]  # number of features
epsilon = 0.001

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
    y_pred_prob_cv = sess.run(tf.nn.softmax(prediction), feed_dict={X: X_cv})

# ------------------ ROC ------------------
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_cv_enc[:, i], y_pred_prob_cv[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_cv_enc.ravel(), y_pred_prob_cv.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area; first aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.6f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.6f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'b', 'g',
#                 'r', 'c', 'm', 'y'])
# for i, color in zip(range(num_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics of all classes')
plt.legend(loc="lower right")
plt.savefig('D:/kaggle-otto/roc/roc_ann.png')
plt.close()
