import rospy
from pfvtr.msg import DistancedTwist, MapRepeaterAction, MapRepeaterResult, MapRepeaterGoal, SensorsInput, SensorsOutput
from pfvtr.srv import StopRepeater
import actionlib
import numpy as np
from abc import ABC, abstractmethod
import rosbag
from pfvtr.msg import FeaturesList, ImageList, Features, SensorsInput, Histogram
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Quaternion
from sensor_msgs.msg import Image
import copy
import os

class Processing:
    def __init__(self):
        self._reinit()
        self.map_publish_span = 2
        self.sensors_pub = rospy.Publisher("/pfvtr/map_representations", SensorsInput, queue_size=1)

    def _reinit(self):
        self.map_images = []
        self.map_distances = []
        self.map_times = []
        self.map_alignments = []
        self.map_transitions = []
        self.nearest_map_img = 0
        self.map_num = 1
        self.curr_map = 0
        self.last_map = 0

    def _load_map(self, mappaths, images, distances, trans, times, source_align):
        if "," in mappaths:
            mappaths = mappaths.split(",")
        else:
            mappaths = [mappaths]
        for map_idx, mappath in enumerate(mappaths):
            tmp = []
            for file in list(os.listdir(mappath)):
                if file.endswith(".npy"):
                    tmp.append(file[:-4])
            rospy.logwarn(str(len(tmp)) + " images found in the map")
            tmp.sort(key=lambda x: float(x))
            tmp_images = []
            tmp_distances = []
            tmp_trans = []
            tmp_times = []
            tmp_align = []
            source_map = None

            for idx, dist in enumerate(tmp):
                tmp_distances.append(float(dist))
                with open(os.path.join(mappath, dist + ".npy"), 'rb') as fp:
                    map_point = np.load(fp, allow_pickle=True, fix_imports=False).item(0)
                    r = map_point["representation"]
                    ts = map_point["timestamp"]
                    diff_hist = map_point["diff_hist"]
                    # this logic is for using multiple map - all maps must have same source map
                    if map_point["source_map_align"] is None:
                        sm = mappath
                    else:
                        sm = map_point["source_map_align"][0]
                    if source_map is None:
                        source_map = sm
                    if sm != source_map:
                        rospy.logwarn("Multimap with invalid target!" + str(mappath))
                        raise Exception("Invalid map combination")
                    if map_point["source_map_align"] is not None:
                        align = map_point["source_map_align"][1]
                    else:
                        align = 0
                    feature = Features()
                    feature.shape = r[0].shape
                    feature.values = list(r[0].flatten())
                    feature.descriptors = r[1]
                    tmp_images.append(feature)
                    tmp_times.append(ts)
                    tmp_align.append(align)
                    if diff_hist is not None:
                        tmp_trans.append(diff_hist)
                    # rospy.loginfo("Loaded feature: " + dist + str(".npy"))
            tmp_times[-1] = tmp_times[-2] + (tmp_times[-2] - tmp_times[-3])  # to avoid very long period before map end
            images.append(tmp_images)
            distances.append(tmp_distances)
            trans.append(tmp_trans)
            times.append(tmp_times)
            source_align.append(tmp_align)
            rospy.logwarn("Whole map " + str(mappath) + " sucessfully loaded")

    def pubSensorsInput(self, distance):
        # rospy.logwarn("Obtained image!")
        if len(self.map_images) > 0:
            # rospy.logwarn(self.map_distances)
            # Load data from each map the map
            features = []
            distances = []
            timestamps = []
            offsets = []
            transitions = []
            last_nearest_img = self.nearest_map_img
            map_indices = []
            for map_idx in range(self.map_num):
                self.nearest_map_img = np.argmin(abs(distance - np.array(self.map_distances[map_idx])))
                # allow only move in map by one image per iteration
                lower_bound = max(0, self.nearest_map_img - self.map_publish_span)
                upper_bound = min(self.nearest_map_img + self.map_publish_span + 1, len(self.map_distances[map_idx]))

                features.extend(self.map_images[map_idx][lower_bound:upper_bound])
                distances.extend(self.map_distances[map_idx][lower_bound:upper_bound])
                timestamps.extend(self.map_times[map_idx][lower_bound:upper_bound])
                offsets.extend(self.map_alignments[map_idx][lower_bound:upper_bound])
                transitions.extend(self.map_transitions[map_idx][lower_bound:upper_bound - 1])
                map_indices.extend([map_idx for i in range(upper_bound - lower_bound)])
            if self.nearest_map_img != last_nearest_img:
                rospy.loginfo("matching image " + str(self.map_distances[-1][self.nearest_map_img]) +
                              " at distance " + str(distance))
            transitions = np.array(transitions)
            # Create message for estimators
            sns_in = SensorsInput()
            sns_in.header.stamp = rospy.Time.now()
            sns_in.live_features = []
            sns_in.map_features = features
            sns_in.map_distances = distances
            sns_in.map_transitions = [Histogram(transitions.flatten(), transitions.shape)]
            sns_in.map_timestamps = timestamps
            sns_in.map_num = self.map_num
            # TODO: sns_in.map_similarity
            sns_in.map_offset = offsets

            # rospy.logwarn("message created")
            self.sensors_pub.publish(sns_in)
            self.last_map = self.curr_map

    def load_map(self, map_name):
        self._reinit()
        self._load_map(map_name, self.map_images, self.map_distances,
                       self.map_transitions, self.map_times, self.map_alignments)

