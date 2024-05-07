from pfvtr.msg import SensorsOutput, ImageList, Features
import rospy
import numpy as np
from pfvtr.msg import FeaturesList, ImageList, Features, SensorsInput, Histogram
from collections import deque


class DataFetching:
    def __init__(self, input_size):
        self.header = None
        self.last_time = None
        self.input_size = input_size
        self.map_num = 1
        self.features_sub = rospy.Subscriber("/pfvtr/matched_repr", SensorsInput, self.processing_callback, queue_size=1)
        self.data = []
        self.has_new_obs = False

    def processing_callback(self, msg):
        curr_time = msg.header.stamp
        self.header = msg.header
        if self.last_time is None:
            self.last_time = curr_time
            return
        hists = np.array(msg.map_histograms[0].values).reshape(msg.map_histograms[0].shape)
        self.last_hists = hists
        map_trans = np.array(msg.map_transitions[0].values).reshape(msg.map_transitions[0].shape)
        live_hist = np.array(msg.live_histograms[0].values).reshape(msg.live_histograms[0].shape)
        hist_width = hists.shape[-1]
        shifts = np.round(np.array(msg.map_offset) * (hist_width // 2)).astype(int)
        hists = np.roll(hists, shifts, -1)  # not sure if last dim should be rolled like this
        dists = np.array(msg.map_distances)
        timestamps = msg.map_timestamps

        # Divide incoming data according the map affiliation
        len_per_map = np.size(dists) // self.map_num
        trans_per_map = len_per_map - 1
        if len(dists) % msg.map_num > 0:
            # TODO: this assumes that there is same number of features comming from all the maps (this does not have to hold when 2*map_len < lookaround)
            rospy.logwarn("!!!!!!!!!!!!!!!!!! One map has more images than other !!!!!!!!!!!!!!!!")
            return
        map_trans = [map_trans[trans_per_map * map_idx:trans_per_map * (map_idx + 1)] for map_idx in
                     range(self.map_num)]
        hists = [hists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
        dists = np.array([dists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)])
        timestamps = [timestamps[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
        time_diffs = [self._get_time_diff(timestamps[map_idx]) for map_idx in range(self.map_num)]

        in_diff = self.input_size - hists[0].shape[0]
        if in_diff > 0:
            in_append = np.zeros((in_diff, 1024))
            if dists[0][0] < 2.0:
                out_hists = np.concatenate((in_append, hists[0]), axis=0)
                out_map_trans = np.concatenate((in_append, map_trans[0]), axis=0)
            else:
                out_hists = np.concatenate((hists[0], in_append), axis=0)
                out_map_trans = np.concatenate((map_trans[0], in_append), axis=0)
        else:
            out_hists = hists[0]
            out_map_trans = map_trans[0]
        self.data = (dists, out_hists, out_map_trans, np.expand_dims(live_hist, axis=0))
        self.has_new_obs = True

    def get_live_data(self):
        if len(self.data) > 0 and self.has_new_obs:
            self.has_new_obs = False
            return self.data
        else:
            return None

    def _get_time_diff(self, timestamps: list):
        out = []
        for i in range(len(timestamps) - 1):
            out.append((timestamps[i + 1] - timestamps[i]).to_sec())
        return out