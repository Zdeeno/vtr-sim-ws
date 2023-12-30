import rospy
import actionlib
import rosbag
import numpy as np
import os
from geometry_msgs.msg import Twist, Pose, Quaternion
from nav_msgs.msg import Odometry
from pfvtr.msg import DistancedTwist, MapRepeaterAction, MapRepeaterResult, MapRepeaterGoal, SensorsInput, SensorsOutput
from visualization_msgs.msg import MarkerArray, Marker
import matplotlib.pyplot as plt
import copy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from vtr import PFVTR

plt.switch_backend('TKAgg')

MAP_DIR = "/home/zdeeno/.ros/simulator_maps"
USE_VTR = True

class Map:
    def __init__(self, map_dir):
        rospy.logwarn("Parsin map at: " + map_dir)
        self.name = map_dir.split("/")[-1]
        self.bag_path = os.path.join(map_dir + "/" + self.name + ".bag")
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
    def __init__(self, map_dir, use_vtr):        
        # visualization initialization
        self.use_vtr = use_vtr
        self.target_dist = 0.0

        self.plot_wait = 100
        self.plot_counter = self.plot_wait
        
        self.curr_x = None
        self.curr_y = None
        
        # ros initialization
        rospy.init_node('simulator')
        self.marker_pub = rospy.Publisher("/map_viz", MarkerArray, queue_size=5)
        self.teleport_pub = rospy.Publisher("/robot1/chassis_link/teleport", Pose)
        
        # parsing maps
        all_map_dirs = [ f.path for f in os.scandir(map_dir) if f.is_dir() ]
        self.maps = [Map(p) for p in all_map_dirs]
        
        # sub gt position
        self.pos_sub = rospy.Subscriber("/robot1/odometry", Odometry, self.odom_callback)
 
    def reset_sim(self):
        # clear variables
        rospy.logwarn("Starting new round!")
        self.trav_x = []
        self.trav_y = []
        self.init_pos = None
        plt.close()
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        # choose new map
        new_map_idx = np.random.randint(len(self.maps))
        # plot map and teleport the robot
        self.plt_trajectory(new_map_idx)
        self.target_dist = self.maps[new_map_idx].dists[-1]
        dist = self.teleport(new_map_idx, pos_err=2.0, rot_err=0.5)
        rospy.sleep(1) # avoid plotting errors
        return new_map_idx, dist

    def plt_robot(self):
        # TODO: Throw away trav coords for long traversals
        if self.curr_x is not None and self.curr_y is not None:
            self.trav_x.append(self.curr_x)
            self.trav_y.append(self.curr_y)
        self.plot_counter += 1
        if self.plot_wait <= self.plot_counter:
            if self.init_pos is not None:
                self.ax.scatter(self.init_pos[0], self.init_pos[1], marker="x", color="r")
                self.init_pos = None
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
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y

    def teleport(self, map_idx, pos_err=0.0, rot_err=0.0):
        pct_dist = 0.5 * np.random.rand(1)[0]
        starting_dist_idx = int(len(self.maps[map_idx].odoms) * pct_dist)
        odom_pos = copy.copy(self.maps[map_idx].odoms[starting_dist_idx])
        self.init_pos = (odom_pos.pose.pose.position.x, odom_pos.pose.pose.position.y)
        pose_to = Pose()
        diff_x, diff_y, diff_phi = np.random.rand(3) * 2.0 - 1.0
        # randomize the spawning
        pose_to.position.x = odom_pos.pose.pose.position.x + diff_x * pos_err       
        pose_to.position.y = odom_pos.pose.pose.position.y + diff_y * pos_err
        a, b, c = euler_from_quaternion([odom_pos.pose.pose.orientation.x, odom_pos.pose.pose.orientation.y, odom_pos.pose.pose.orientation.z, odom_pos.pose.pose.orientation.w])
        target_quat = quaternion_from_euler(a, b, c + diff_phi * rot_err)
        pose_to.orientation = Quaternion(*target_quat)
        pose_to.position.z = odom_pos.pose.pose.position.z + 0.25 # spawn bit higher to avoid textures        
        self.teleport_pub.publish(pose_to)
        rospy.logwarn("Teleporting robot!")
        return self.maps[map_idx].dists[starting_dist_idx]
        
    def vtr_traversal(self, vtr, map_idx, start_pos):
        rospy.logwarn("Traversing using VTR at distance " + str(start_pos))
        map_name = self.maps[map_idx].name
        end_pos = self.maps[map_idx].dists[-1]
        rospy.loginfo("Starting traversal of map: " + map_name)
        
        vtr.repeat_map(start_pos, end_pos, map_name)
        
        return
    

class Environment:
    def __init__(self, map_dir, use_vtr):
        # subscribe observations
        self.sim = None
        
        # start simulation
        self.sim = Simulator(map_dir, use_vtr)
        self.vtr = PFVTR()
        
        while True:
            # main simulation loop
            map_idx, dist = self.sim.reset_sim()
            rospy.sleep(2)
            self.sim.vtr_traversal(self.vtr, map_idx, dist)
            while not self.vtr.is_finished():
                self.sim.plt_robot()
            rospy.sleep(2)
            
    

if __name__ == '__main__':
    sim = Environment(MAP_DIR, USE_VTR)
