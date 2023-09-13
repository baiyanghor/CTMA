import math

import numpy as np
from scipy.stats import relfreq
from SFMOperations.VGGFunctions import X_from_xP_nonlinear
from .SFMDataTypes import *


class NViewCalculator(object):
    def __init__(self, in_pipeline_data: GlobalSFMData):
        self.sfm_pipeline_data = in_pipeline_data

    def construct_trajectory(self):
        pass

    def select_initial_triangulation_pair(self):
        # Select by distribution of key points
        height = self.sfm_pipeline_data.sequence_image_size[0]
        width = self.sfm_pipeline_data.sequence_image_size[1]
        x_bins = 16
        y_bins = math.floor(x_bins * height / width)

        distribution_list = []
        for a_image in self.sfm_pipeline_data.sfm_images:
            match_pre = a_image.dump_matches_to_pre()
            match_next = a_image.dump_matches_to_next()

            if match_pre.shape[0] > 0:
                x_frequency_pre = relfreq(match_pre[:,0], x_bins, defaultreallimits=(1, width)).frequency
                y_frequency_pre = relfreq(match_pre[:,1], y_bins, defaultreallimits=(1, height)).frequency
                x_var_pre = np.std(x_frequency_pre)
                y_var_pre = np.std(y_frequency_pre)
            else:
                x_var_pre = 0
                y_var_pre = 0

            if match_next.shape[0] > 0:
                x_frequency_next = relfreq(match_next[:,0], x_bins, defaultreallimits=(1, width)).frequency
                y_frequency_next = relfreq(match_next[:,1], y_bins, defaultreallimits=(1, height)).frequency
                x_var_next = np.std(x_frequency_next)
                y_var_next = np.std(y_frequency_next)
            else:
                x_var_next = 0
                y_var_next = 0

            total_var = np.std([x_var_pre, y_var_pre, x_var_next, y_var_next])
            distribution_list.append(total_var)

        selected_image_idx = np.argmin(np.array(distribution_list))

        if 0 < selected_image_idx < self.sfm_pipeline_data.cached_image_count-1:
            if distribution_list[selected_image_idx - 1] > distribution_list[selected_image_idx + 1]:
                return selected_image_idx, selected_image_idx+1
            else:
                return selected_image_idx-1, selected_image_idx

        elif selected_image_idx == 0:
            return selected_image_idx, selected_image_idx + 1

        elif selected_image_idx == self.sfm_pipeline_data.cached_image_count-1:
            return selected_image_idx - 1, selected_image_idx



