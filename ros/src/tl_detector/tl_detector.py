#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import tensorflow
import keras

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoint_tree = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        # might want to use image_raw instead for more data
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        graph = self.load_graph('/capstone/ros/src/tl_detector/saved_models/traffic_light_classifier.pb')

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(graph)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.last_car_wp = 0
        self.state_count = 0
        self.img_idx = 0

        # model = keras.models.load_model('/capstone/ros/src/tl_detector/traffic_light_classifier.h5')

        rospy.spin()

    def load_graph(self, frozen_graph_filename):
        # Source: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tensorflow.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tensorflow.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tensorflow.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tensorflow.import_graph_def(graph_def, name="prefix")
        return graph

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # rospy.logwarn("Closest light wp: {0}\nAnd light state: {1}".format(light_wp, state))

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            # Stop on red or yellow light just to be safe
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            x (Pose position x): x position to match a waypoint to
            y (Pose position y): y position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light, line_wp_idx, car_wp_idx):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        state = TrafficLight.UNKNOWN

        if not self.has_image:
            self.prev_light_loc = None
            return False

        # rospy.logwarn("Next line wp: {}, Car Wp: {}".format(line_wp_idx, car_wp_idx))
        if line_wp_idx - car_wp_idx < 50:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            state = self.light_classifier.get_classification(cv_image)

        # ## Code below used to capture training image dataset.
        # if self.last_car_wp == 0:
        #     self.last_car_wp = car_wp_idx
        # elif self.last_car_wp != car_wp_idx:
        # #     rospy.logwarn("car_wp_idx: {}".format(car_wp_idx))
        #     self.last_car_wp = car_wp_idx
        #     if line_wp_idx - car_wp_idx < 100:
        #         cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #         cv2.imwrite('dataset/{}-{}-{}.png'.format(self.img_idx, line_wp_idx, light.state), cv_image)
        #         rospy.loginfo('Writing image to dataset/{}-{}-{}.png'.format(self.img_idx, line_wp_idx, light.state))
        #         self.img_idx += 1

        # #Get classification
        # for testing, just return the light state
        rospy.loginfo("State: {}".format(state))
        return state
        # return light.state
        # return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light, line_wp_idx, car_wp_idx)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
