import rospy
import rosbag
import numpy as np
import os
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from pfvtr.msg import DistancedTwist
from visualization_msgs.msg import MarkerArray, Marker
import matplotlib.pyplot as plt

plt.switch_backend('TKAgg')

MAP_DIR = "/home/zdeeno/.ros/simulator_maps"

class Map:
    def __init__(self, map_dir):
        rospy.logwarn("Parsin map at: " + map_dir)
        self.name = map_dir.split("/")[-1]
        self.bag_path = os.path.join(map_dir + "/" + self.name + ".bag")
        rospy.logwarn(self.bag_path)
        self.bag = rosbag.Bag(self.bag_path, "r")
        self.dists, self.odoms, self.actions = self.fetch_map(self.bag)
        
    def fetch_map(self, bag):
        dists = []
        actions = []
        odoms = []
        last_odom = None
        for topic, msg, t in self.bag.read_messages(topics=["/recorded_actions", "/recorded_odometry"]):
            if topic == "/recorded_odometry":
                last_odom = msg
            if topic == "/recorded_actions" and last_odom is not None: 
                dists.append(float(msg.distance))
                actions.append(msg.twist)
                odoms.append(last_odom)
        dists = np.array(dists)
        rospy.logwarn("Map " + self.name + " fetched.")
        return dists, odoms, actions
        
    def get_rviz_marker(self):
        m_arr = MarkerArray()
        for i in range(len(self.odoms)):
            marker = Marker()
            marker.header.frame_id = "map"
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
            marker.pose.position = self.odoms[i].pose.pose.position
            marker.pose.orientation = self.odoms[i].pose.pose.orientation
            m_arr.markers.append(marker)
        return m_arr

    def get_plt_array(self):
        x = []
        y = []
        for i in range(len(self.odoms)):
            x.append(self.odoms[i].pose.pose.position.x)
            y.append(self.odoms[i].pose.pose.position.y)
        return x, y


class Simulator:
    def __init__(self):
        # visualization initialization
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.trav_x = []
        self.trav_y = []
        self.plot_wait = 100
        self.plot_counter = self.plot_wait
        
        # ros initialization
        rospy.init_node('simulator')
        self.marker_pub = rospy.Publisher("/map_viz", MarkerArray, queue_size=5)
        self.teleport_pub = rospy.Publisher("/robot1/chassis_link/teleport", Pose)

        # parsing maps
        all_map_dirs = [ f.path for f in os.scandir(MAP_DIR) if f.is_dir() ]
        self.maps = [Map(p) for p in all_map_dirs]

        # plot map and teleport robot
        self.plt_trajectory(0)
        self.teleport(0)
        
        # sub gt position
        self.pos_sub = rospy.Subscriber("/robot1/odometry", Odometry, self.odom_callback)
        
        # run simulation loop
        self.main_loop()

        rospy.spin()  
 
    def main_loop(self):
        # Main simulation loop
        while True:
            self.plt_robot()

    def plt_robot(self):
        # TODO: Throw away trav coords for long traversals
        self.plot_counter += 1
        if self.plot_wait <= self.plot_counter: # does it work? does not seem waiting ...
            self.ax.plot(self.trav_x, self.trav_y, "r")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.plot_counter = 0
    
    def rviz_trajectory(self, map_idx):
        marker_pub.publish(self.maps[map_idx].get_rviz_marker())
        rospy.logwarn("Markers published!")

    def plt_trajectory(self, map_idx):
        x, y = self.maps[map_idx].get_plt_array()
        self.ax.plot(x, y)
        plt.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def odom_callback(self, msg):
        self.trav_x.append(msg.pose.pose.position.x)
        self.trav_y.append(msg.pose.pose.position.y)

    def teleport(self, map_idx):
        odom_pos = self.maps[map_idx].odoms[0]
        pose_to = odom_pos.pose.pose
        pose_to.position.z += 0.5 # spawn bit higher to avoid textures
        self.teleport_pub.publish(pose_to)
        rospy.logwarn("Teleporting robot!")

if __name__ == '__main__':
    sim = Simulator()
