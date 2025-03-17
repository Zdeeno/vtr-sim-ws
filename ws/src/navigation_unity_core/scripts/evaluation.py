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
from vtr import PFVTR, InformedVTR, NeuralNet2, RLAgent
from navigation_unity_msgs.srv import ResetWorld, ResetWorldRequest, ResetWorldResponse
from world_generator import WorldGenerator
import yaml
import time
from simulator import Simulator


class BasinEvaluator:
    def __init__(self, simulator, vtr, map):
        # subscribe observations
        self.map_idx = None
        self.failure = None
        self.dist = None
        self.sim = None
        self.map_name = None
        self.traversal_idx = 0
        self.force_map = map

        self.sim = simulator
        self.vtr = vtr

        self.basin_grid = []
        arr = np.linspace(-3.0, 3.0, 10)
        for x in arr:
            for y in arr:
                self.basin_grid.append((x, y))

    def round_setup(self, day_time=None, scene=None, random_teleport=None):
        rospy.logwarn("------------ Starting round " + str(self.traversal_idx) + "! --------------")
        self.traversal_idx += 1

        self.failure = False
        disp = self.basin_grid[self.traversal_idx - 1]
        rospy.logwarn("Evaluating displacement: " + str(disp))
        self.map_idx, self.map_name, self.dist = self.sim.reset_sim(day_time, scene, random_teleport,
                                                                    displacement=(disp[0], disp[1], 0.0),
                                                                    force_map_idx=self.force_map)
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
            self.sim.traversal_summary(save=True, idx=self.traversal_idx)
        self.vtr.reset()
        self.sim.plt_robot(save_fig=True, idx=self.traversal_idx)
        self.sim.control_pub.publish(Twist())  # stop robot movement traversing
        time.sleep(3)

    def test_setups(self):
        self.round_setup(0.5, 0)
        # time.sleep(30)
        # self.round_setup(12, 1)
        # time.sleep(30)


if __name__ == '__main__':

    HOME = os.path.expanduser('~')
    MAP_DIR = HOME + "/.ros"
    MAP_IDX = 1

    # start simulation
    simulator = Simulator(MAP_DIR, pose_err_weight=1.0, rot_err_weight=np.pi / 16.0,
                          dist_weight=0.5)
    # informed policy for benchmarking
    # vtr = InformedVTR()

    # PFVTR policy
    # vtr = PFVTR(image_pub=2)

    # Neural network controller
    vtr = RLAgent()

    sim = BasinEvaluator(simulator, vtr, MAP_IDX)
    day_time = 0.0  # daylight between 0.21 to 0.95
    # sim.test_setups()
    while True:
        pass
        sim.round_setup(day_time=np.random.uniform(0.3, 0.7), scene=0, random_teleport=True)
        sim.simulation_forward()