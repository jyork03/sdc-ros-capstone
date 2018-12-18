import rospy
from styx_msgs.msg import TrafficLight
# from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self, graph):
        #TODO load classifier
        self.graph = graph
        # self.model = model
        # for op in graph.get_operations():
        #     print(op.name)
        rospy.loginfo('Getting Graph tensors')
        self.input = graph.get_tensor_by_name('prefix/conv2d_1_input:0')
        self.output = graph.get_tensor_by_name('prefix/dense_2/Softmax:0')
        self.lf = graph.get_tensor_by_name('prefix/keras_learning_phase:0')
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = cv2.resize(image, (100, 75))
        img = np.expand_dims(image,0)

        # predictions = self.model.predict(img)
        # return np.argmax(predictions[0])
        # with tf.Session(graph=self.graph) as sess:
        out = self.sess.run(self.output, feed_dict={self.input: img, self.lf: 0})

        return np.argmax(out)
