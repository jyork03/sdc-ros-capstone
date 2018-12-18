import glob
import cv2
from sklearn.model_selection import train_test_split
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def get_labeled_images(path):
    image_paths = np.array(glob.glob(path))
    images = []
    labels = []
    for path in image_paths:
        # read the image, and make sure it's RGB
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        # normalize the image to a range of 0-1
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = cv2.resize(image, (100, 75))
        images.append(image)
        # get the label from the filename
        label = int(re.match('.+([0-9])[^0-9]*$', path).group(1))
        labels.append(label)
        # X_train, X_test, y_train, y_test
    return train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)

# helper function for viewing images in pyplot columns
def side_by_side(images, labels=("", "", ""), cmap='viridis', cols=3):
    f, axes = plt.subplots(1, cols, figsize=(20,10))
    cmapv = cmap
    for idx in range(cols):

        if isinstance(cmap, str) is False:
            cmapv = cmap[idx]

        axes[idx].imshow(images[idx], cmap=cmapv)
        axes[idx].set_title(labels[idx], fontsize=30)

def one_hot_encode(x):
    # One hot encode the labels {0, 1, 2}
    one_hot = np.zeros((len(x), 3))
    for i, val in enumerate(x):
        one_hot[i][val] = 1
    
    return one_hot

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    # Source: https://stackoverflow.com/a/45466355/3149695
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def load_graph(frozen_graph_filename):
    # Source: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph