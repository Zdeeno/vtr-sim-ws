import rospy
from pfvtr.msg import DistancedTwist, MapRepeaterAction, MapRepeaterResult, MapRepeaterGoal, SensorsInput, SensorsOutput
import actionlib
import numpy as np

class PFVTR:
    def __init__(self):
        rospy.logwarn("Trying to instantiate PFCTR class")
        
        self.map_hists = None  # histogram comparing live vs map images
        self.live_diff_hist = None  # curr live img vs prev live img 
        self.dists = None  # distances of map images 
        self.map_diff_hist = None  # consecutive map images comparison
        self.est_dist = 0.0
        self.finished = True
        self.target_dist = 9999999999
        self.map_start = False
        
        self.obs_sub = rospy.Subscriber("/pfvtr/matched_repr", SensorsInput, self._obs_callback, queue_size=1)
        self.distance_sub = rospy.Subscriber("/pfvtr/repeat/output_dist", SensorsOutput, self._dist_callback, queue_size=1)
        
        self.client = actionlib.SimpleActionClient("/pfvtr/repeater", MapRepeaterAction) # for VTR
        rospy.logwarn("PFVTR successfully connected!")
        
    def repeat_map(self, start, end, map_name):
        """
        Required method!
        """
        self.finished = False
        self.target_dist = end
        self.map_start = True
        curr_action = MapRepeaterGoal(startPos=start, endPos=end, traversals=0, nullCmd=True, imagePub=1, useDist=True, mapName=map_name)
        self.client.send_goal(curr_action)
        return
    
    def is_finished(self):
        return self.finished
    
    def _obs_callback(self, msg):
        # fetch all possible observations
        self.map_hists = np.array(msg.map_histograms[0].values).reshape(msg.map_histograms[0].shape)
        self.live_diff_hist = np.array(msg.live_histograms[0].values).reshape(msg.live_histograms[0].shape)
        self.dists = np.array(msg.map_distances)
        self.map_diff_hist = np.array(msg.map_transitions[0].values).reshape(msg.map_transitions[0].shape)
        
    def _dist_callback(self, msg):
        # fetch estimated distance
        if self.map_start:
            rospy.sleep(3)
            self.map_start = False
            return
        self.est_dist = msg.output
        if self.est_dist >= self.target_dist - 0.5:
            self.finished = True
        
    

