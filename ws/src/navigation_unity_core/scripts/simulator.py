import rospy
import actionlib
import rosbag
import rospkg
import numpy as np
import os
from geometry_msgs.msg import Twist, Pose, Quaternion
from nav_msgs.msg import Odometry
from pfvtr.msg import DistancedTwist, MapRepeaterAction, MapRepeaterResult, MapRepeaterGoal, SensorsInput, SensorsOutput
from visualization_msgs.msg import MarkerArray, Marker
import matplotlib.pyplot as plt
import copy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from vtr import PFVTR, InformedVTR, NeuralNet
from navigation_unity_msgs.srv import ResetWorld, ResetWorldRequest, ResetWorldResponse
from world_generator import WorldGenerator
import yaml
import time

np.random.seed(17)
plt.switch_backend('TKAgg')
home = os.path.expanduser('~')
scenes = ["forest", "mall"]

HOME = os.path.expanduser('~')
MAP_DIR = HOME + "/.ros"


class Map:
    def __init__(self, map_dir):
        rospy.logwarn("Parsin map at: " + map_dir)
        self.name = map_dir.split("/")[-1]
        self.bag_path = os.path.join(map_dir + "/" + self.name + ".bag")
        self.bag = rosbag.Bag(self.bag_path, "r")
        self.dists, self.odoms, self.actions = self.fetch_map()

    def fetch_map(self):
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
        rospy.logwarn("Map " + self.name + " fetched with " + str(len(dists)) + " points " + str(
            len(actions)) + " actions " + str(len(odoms)) + " odometries")
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
        phi = []
        for i in range(len(self.odoms)):
            x.append(self.odoms[i].pose.pose.position.x)
            y.append(self.odoms[i].pose.pose.position.y)
            _, _, c = euler_from_quaternion([self.odoms[i].pose.pose.orientation.x,
                                             self.odoms[i].pose.pose.orientation.y,
                                             self.odoms[i].pose.pose.orientation.z,
                                             self.odoms[i].pose.pose.orientation.w])
            phi.append(c)
        return x, y, phi


class Simulator:
    def __init__(self, map_dir, pose_err_weight=1.0, rot_err_weight=np.pi / 4.0, dist_weight=0.5):
        self.failure_dist = 5.0
        self.failure_angle = np.pi / 2.0
        self.target_dist = 0.0
        self.not_moving_trsh = rospy.Duration(5)
        self.last_moved_time = None
        self.moving_dist = 1.0

        self.plot_wait = 100
        self.plot_counter = self.plot_wait
        
        self.curr_x = None
        self.curr_y = None
        self.last_x = None
        self.last_y = None
        self.curr_phi = None
        self.curr_map_idx = None
        self.init_displacement = 0.0
        self.init_rot_error = 0.0
        self.init_pos = None

        self.map_traj = None
        # might be wrong:::::
        self.teleport_displacement = pose_err_weight
        self.teleport_rotation = rot_err_weight
        self.spawn_error_weight_pose = self.teleport_displacement
        self.spawn_error_weight_rot = self.teleport_rotation
        self.spawn_distance_weight = dist_weight
        self.map_phis = None
        
        # ros initialization
        rospy.init_node('simulator')
        rospy.wait_for_service('reset_world')
        rospy.logwarn("Change world service available.")
        self.marker_pub = rospy.Publisher("/map_viz", MarkerArray, queue_size=5)
        self.teleport_pub = rospy.Publisher("/robot1/chassis_link/teleport", Pose, queue_size=1)
        self.control_pub = rospy.Publisher("/bluetooth_teleop/cmd_vel", Twist, queue_size=1)
        self.reset_world = rospy.ServiceProxy('reset_world', ResetWorld)

        # parsing maps
        all_map_dirs = [f.path for f in os.scandir(map_dir) if f.is_dir() and "vtr" in f.path]
        self.maps = [Map(p) for p in all_map_dirs]
        rospack = rospkg.RosPack()
        self.world_path = rospack.get_path('navigation_unity_core') + "/unity_world_config/"
        self.avail_worlds = ["default_world.yaml", "rainy_day.yaml"]
        self.last_world = None
        
        # sub gt position
        ## init generator
        self.world_path = rospack.get_path('navigation_unity_core') + "/unity_world_config/"
        self.world_generator = WorldGenerator(service=self.reset_world, world_path=self.world_path, scenes=scenes)
        self.pos_sub = rospy.Subscriber("/robot1/odometry", Odometry, self.odom_callback, queue_size=1)

    def reset_sim(self, day_time=None, scene=None, teleport=None):
        # clear variables
        rospy.logwarn("Starting new round!")
        self.randomly_change_world()
        self.trav_x = []
        self.trav_y = []
        self.map_traj = None
        self.init_pos = None
        plt.close()
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        rospy.logwarn("Starting new round!")
        spawn_point = None
        # change world
        if day_time is not None and scene is not None:
            day_minutes, day_hours, day_progress_speed, fog_density, spawn_point = self.world_generator.time_based_change_world(
                day_time, scene, teleport, fog_density=0.0, lights=False)
            rospy.logwarn("Time based change to " + str(day_hours) + ":" + str(day_minutes) + " with speed " + str(
                day_progress_speed) + " and fog " + str(fog_density) + " and spawn point " + str(spawn_point))
            new_map_idx = scene

        else:
            scene = self.world_generator.randomly_change_world()
            new_map_idx = np.random.randint(len(self.maps))

        self.curr_map_idx = new_map_idx
        # plot map and teleport the robot
        self.plt_trajectory(self.curr_map_idx)
        self.target_dist = self.maps[self.curr_map_idx].dists[-1]
        if teleport is None or not teleport:
            dist = 0.0
        else:
            dist = self.teleport(self.maps[self.curr_map_idx])
        self.last_moved_time = None
        self.last_x = None
        self.last_y = None
        rospy.sleep(1)  # avoid plotting errors
        return new_map_idx, self.maps[new_map_idx].name, dist

    def plt_robot(self, save_fig=False, idx=0):
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
        if save_fig:
            rospy.logwarn("Saving trajectory image to: " + HOME + "/.ros/trajectory_plots/" + str(idx) + ".jpg")
            self.fig.savefig(HOME + "/.ros/trajectory_plots/" + str(idx) + ".jpg")
    
    def rviz_trajectory(self, map_idx):
        marker_pub.publish(self.maps[map_idx].get_rviz_marker())
        rospy.logwarn("Markers published!")

    def plt_trajectory(self, map_idx):
        map_traj_x, map_traj_y, map_traj_phi = self.maps[map_idx].get_plt_array()
        self.ax.plot(map_traj_x, map_traj_y)
        self.map_traj = np.column_stack((map_traj_x, map_traj_y))
        self.map_phis = np.array(map_traj_phi)
        plt.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        a, b, c = euler_from_quaternion(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w])
        self.curr_phi = c

    def teleport(self, map):
        pct_dist = self.spawn_distance_weight * np.random.rand(1)[0]
        starting_dist_idx = int(len(map.odoms) * pct_dist)
        odom_pos = copy.copy(map.odoms[starting_dist_idx])
        pose_to = Pose()
        diff_x, diff_y, diff_phi = np.random.rand(3) * 2.0 - 1.0

        self.init_pos = (odom_pos.pose.pose.position.x, odom_pos.pose.pose.position.y)
        self.init_displacement = (diff_x, diff_y)
        self.init_rot_error = diff_phi
        # randomize the spawning
        pose_to.position.x = odom_pos.pose.pose.position.x + diff_x * self.spawn_error_weight_pose
        pose_to.position.y = odom_pos.pose.pose.position.y + diff_y * self.spawn_error_weight_pose
        a, b, c = euler_from_quaternion(
            [odom_pos.pose.pose.orientation.x, odom_pos.pose.pose.orientation.y, odom_pos.pose.pose.orientation.z,
             odom_pos.pose.pose.orientation.w])
        target_quat = quaternion_from_euler(a, b, c + diff_phi * self.spawn_error_weight_rot)
        pose_to.orientation = Quaternion(*target_quat)
        pose_to.position.z = odom_pos.pose.pose.position.z + 0.25  # spawn bit higher to avoid textures
        self.teleport_pub.publish(pose_to)
        rospy.logwarn("Teleporting robot to " + str(pose_to.position.x) + " " + str(pose_to.position.y) + " " + str(
            pose_to.position.z))
        return map.dists[starting_dist_idx]

    def vtr_traversal(self, vtr, map_idx, start_pos):
        rospy.logwarn("Traversing using VTR at distance " + str(start_pos))
        map_name = self.maps[map_idx].name
        end_pos = self.maps[map_idx].dists[-1]
        rospy.loginfo("Starting traversal of map: " + map_name)
        vtr.repeat_map(start_pos, end_pos, map_name)
        return
    
    def traversal_summary(self):
        # TODO: use class variables which have precalculated trajectory
        odom_pos = copy.copy(self.maps[self.curr_map_idx].odoms[-1])
        last_x, last_y = (odom_pos.pose.pose.position.x, odom_pos.pose.pose.position.y)
        a, b, last_phi = euler_from_quaternion(
            [odom_pos.pose.pose.orientation.x, odom_pos.pose.pose.orientation.y, odom_pos.pose.pose.orientation.z,
             odom_pos.pose.pose.orientation.w])
        last_phi_err = self.smallest_angle_diff(last_phi, self.curr_phi)
        final_displacement = (self.curr_x - last_x, self.curr_y - last_y)
        rospy.logwarn("\n--------- Summary ---------:\nInit displacement distance/rotation " + str(
            self.init_displacement) + "/" + str(self.init_rot_error) + "\n" +
                      "Final displacement distance/rotation " + str(final_displacement) + "/" + str(
            last_phi_err) + "\n" +
                      "Final Chamfer dist: " + str(self._chamfer_dist()) + "\n---------------------------")

    def smallest_angle_diff(self, angle1, angle2):
        diff = angle2 - angle1
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return np.where(diff < -np.pi, diff + 2 * np.pi, diff)

    def _chamfer_dist(self):
        trav_x = np.array(self.trav_x)
        trav_y = np.array(self.trav_y)
        trav_points = np.column_stack((trav_x, trav_y))

        # Function to calculate the minimum distance from each point in one array to the closest point in another array
        def min_distance(points1, points2):
            distances = []
            for point1 in points1:
                # Calculate Euclidean distances from this point to all points in the other array
                dist = np.sqrt(np.sum((points2 - point1) ** 2, axis=1))
                # Append the minimum distance
                distances.append(np.min(dist))
            return np.mean(distances)

        # Calculate the directed Chamfer Distances
        distance_1_to_2 = min_distance(self.map_traj, trav_points)
        distance_2_to_1 = min_distance(trav_points, self.map_traj)

        # The Chamfer Distance is the average of these two distances
        return (distance_1_to_2 + distance_2_to_1) / 2

    def failure_check(self):
        # checking distance from trajectory
        curr_pos = np.array([self.curr_x, self.curr_y])
        dists = np.sqrt(np.sum((curr_pos - self.map_traj) ** 2, axis=1))
        min_dist_arg = np.argmin(dists)
        rot_diff = self.smallest_angle_diff(self.curr_phi, self.map_phis[min_dist_arg])
        if dists[min_dist_arg] > self.failure_dist or abs(rot_diff) > self.failure_angle:
            rospy.logwarn("!!!TRAVERSAL FAILED - TOO FAR FROM MAP!!!")
            return True
        # checking collision - no movement
        if self.last_x is not None and self.last_y is not None:
            dist = np.sqrt((self.last_x - self.curr_x) ** 2 + (self.last_y - self.curr_y) ** 2)
            if dist > self.moving_dist:
                self.last_x = self.curr_x
                self.last_y = self.curr_y
                if self.last_moved_time is None:
                    rospy.logwarn("Robot movement detected.")
                self.last_moved_time = rospy.get_rostime()
        else:
            self.last_x = self.curr_x
            self.last_y = self.curr_y
        if self.last_moved_time is not None and (rospy.get_rostime() - self.last_moved_time) > self.not_moving_trsh:
            rospy.logwarn("!!!TRAVERSAL FAILED - NOT MOVING FOR " + str(self.not_moving_trsh) + "ms!!!")
            return True
        return False


class Environment:
    def __init__(self, map_dir):
        # subscribe observations
        self.map_idx = None
        self.failure = None
        self.dist = None
        self.sim = None
        self.map_name = None
        self.traversal_idx = 0

        self.sim = simulator
        self.vtr = vtr

    def round_setup(self, day_time=None, scene=None, random_teleport=None):
        self.traversal_idx += 1

        self.failure = False
        self.map_idx, self.map_name, self.dist = self.sim.reset_sim(day_time, scene, random_teleport)
        time.sleep(3)

    def simulation_forward(self):
        # main simulation loop
        self.sim.vtr_traversal(self.vtr, self.map_idx, self.dist)
        while not self.vtr.is_finished():
            self.sim.plt_robot()
            self.failure = self.sim.failure_check()
            if self.failure:
                break
        if self.failure:
            rospy.logwarn("!!! UNSUCCESSFUL TRAVERSAL !!!")
        else:
            self.sim.traversal_summary()
        self.sim.plt_robot(save_fig=True, idx=self.traversal_idx)
        self.vtr.reset()
        self.sim.control_pub.publish(Twist())  # stop robot movement traversing
        time.sleep(2)

    def test_setups(self):
        self.round_setup(0.5, 0)
        # time.sleep(30)
        # self.round_setup(12, 1)
        # time.sleep(30)

if __name__ == '__main__':

    # start simulation
    simulator = Simulator(MAP_DIR, pose_err_weight=1.0, rot_err_weight=np.pi / 4.0,
                          dist_weight=0.5)
    # informed policy for benchmarking
    # vtr = InformedVTR()

    # PFVTR policy
    vtr = PFVTR(image_pub=1)

    # Neural network controller
    # vtr = NeuralNet(training=True)

    sim = Environment(simulator, vtr)
    day_time = 0.0  # daylight between 0.21 to 0.95
    # sim.test_setups()
    while True:
        pass
        sim.round_setup(day_time=np.random.uniform(0.21, 0.95), scene=np.random.randint(2), random_teleport=True)
        sim.simulation_forward()
