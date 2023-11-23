import rospy
import rosbag
import numpy as np
import os
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker

MAP_DIR = "/home/zdeeno/.ros/simulator_maps"

class Map:
    def __init__(self, map_dir, map_name):
        self.name = map_name
        self.bag = rosbag.Bag(os.path.join(map_dir, map_name + ".bag"), "r")
        self.dists, self.odoms, self.actions = self.fetch_map(self.bag)
        rospy.logwarn("Map " + map_name + "succesfully fetched.")
        
    def fetch_map(self, bag):
        dists = []
        actions = []
        odoms = []
        last_odom = None
        for topic, msg, t in self.bag.read_messages(topics=["recorded_actions", "/robot1/odometry"]):
            if topic == "/robot1/odometry":
                last_odom = msg
            if topic == "recorded_actions" and last_odom is not None: 
                dists.append(float(msg.distance))
                actions.append(msg.twist)
                odoms.append(last_odom)
        dists = np.array(dists)
        return dists, odoms, actions
        
    def viz_map(self, publisher):
        m_arr = MarkerArray()
        for i in range(len(odoms)):
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.header.stamp = rospy.Time.now()
            marker.type = 0
            marker.id = i

            # Set the scale of the marker
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            # Set the color
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            # Set the pose of the marker
            marker.pose.position.x = self.odoms[[i].pose.pose.position
            marker.pose.orientation.x = self.odoms[i].pose.pose.orientation
            m_arr.markers.append(marker)
        publisher.publish(m_arr)

class Simulator:
    def __init__(self):
        rospy.init_node('simulator')

        # Subscribe to the /bluetooth_teleop/cmd_vel topic with the Twist message
        rospy.Subscriber('/bluetooth_teleop/cmd_vel', Twist, callback)
        # Publish to the /robot1/velocity_reference topic with the TwistStamped message
        marker_pub = rospy.Publisher("/map_viz", MarkerArray, queue_size=5)

        # Fetch all available maps
        all_map_dirs = [x[0] for x in os.walk(MAP_DIR)]
        self.maps = [Map(os.path.join(MAP_DIR, p), p) for p in all_map_dirs]

        rospy.spin()

    def callback(self, cmd_vel):
        self.pub.publish(cmd_vel)
        
        
        


if __name__ == '__main__':
    sim = Simulator()
