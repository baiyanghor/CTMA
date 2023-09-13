import cv2 as cv
import numpy as np
from numpy import ndarray
from pprint import pprint as pp
from SFMOperations.SFMCalculationNodes import CalNodeBase
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.OpenCVOperators import OpenCVOperators
from typing import List
from SFMOperations.SFMDataTypes import SFMTrajectory, MatchedImagePoint, SFM_OPERATION


class TrackTaggedFeatures(CalNodeBase):
    def __init__(self, session_id='', node_name='TrackTaggedFeatures', sfm_work_dir='', start_image_id=-1,
                 end_image_id=-1):
        super(TrackTaggedFeatures, self).__init__(node_name, session_id, sfm_work_dir)
        self._db_ops = MongodbInterface(sfm_work_dir)
        self._opencv_ops = OpenCVOperators()
        self._start_image_id = start_image_id
        self._end_image_id = end_image_id

    def tracking_tagged_features_forward(self, user_tagged_op_id: str, new_operation=True, sfm_operation_id=''):
        # Get user tagged image points
        user_tagged_image_ids, user_tagged_image_points, user_tagged_world_points = \
            self._db_ops.get_user_tagged_features(self.session_id, sfm_operation_id=user_tagged_op_id)

        # Assert start image id next to user tagged image id
        assert self._start_image_id < self._end_image_id, "Start and end image id not satisfy order"

        assert user_tagged_image_ids[-1] == self._start_image_id - 1, "Start image not next to user tagged image id"
        # Calculate trajectories of user tagged features
        sfm_image_list = self._db_ops.get_sfm_image_list(self.session_id)
        previous_image_mat = cv.imread(sfm_image_list[user_tagged_image_ids[-1]].image_file_name,
                                       flags=cv.IMREAD_GRAYSCALE)

        trajectories: List[SFMTrajectory] = []
        world_point_count = user_tagged_world_points.shape[0]
        match_point_list = np.zeros((self._end_image_id - self._start_image_id, world_point_count, 2))
        of_error_set = np.zeros((self._end_image_id - self._start_image_id, world_point_count))

        previous_match_points = np.float32(user_tagged_image_points[1])

        for image_id in range(self._start_image_id, self._end_image_id):
            next_image_mat = cv.imread(sfm_image_list[image_id].image_file_name, flags=cv.IMREAD_GRAYSCALE)

            features_in_next, validates, dis_err = self._opencv_ops.optical_track_features(
                previous_match_points, None, previous_image_mat, next_image_mat)

            if np.count_nonzero(validates[:, 0] == 1) < world_point_count:
                raise AssertionError(f"Not enough tracking points in image {image_id}")
            of_error_set[image_id - self._start_image_id] = dis_err.flatten()
            match_point_list[image_id - self._start_image_id] = features_in_next.copy()
            previous_image_mat = next_image_mat.copy()
            previous_match_points = features_in_next.copy()

        wp_majored_match_point_list = np.transpose(match_point_list, (1, 0, 2))

        im_match_pt_count = self._end_image_id - self._start_image_id + 2
        for_trajectory_construct = np.empty((world_point_count, im_match_pt_count, 2))
        for i in range(world_point_count):
            row = np.hstack((user_tagged_image_points[0][i].reshape(2, 1),
                             user_tagged_image_points[1][i].reshape(2, 1),
                             wp_majored_match_point_list[i].T,))
            for_trajectory_construct[i] = row.T

        # print(f"for_trajectories_construct {for_trajectory_construct.shape}\n{for_trajectory_construct}")
        # print(f"user_tagged_image_points[0] {user_tagged_image_points[0].shape}\n{user_tagged_image_points[0]}")
        # print(f"user_tagged_image_points[1] {user_tagged_image_points[1].shape}\n{user_tagged_image_points[1]}")
        # print(f"wp_majored_match_point_list {wp_majored_match_point_list.shape}\n{wp_majored_match_point_list}")

        for wp, im_points in zip(user_tagged_world_points, for_trajectory_construct):
            new_trajectory = SFMTrajectory()
            new_trajectory.start_image_id = self._start_image_id - 2
            new_trajectory.traject_length = self._end_image_id - self._start_image_id + 2
            new_trajectory.world_point = wp
            for a_pos, image_id in zip(im_points, range(self._start_image_id - 2, self._end_image_id)):
                new_match_point = MatchedImagePoint()
                new_match_point.sfm_image_id = image_id
                new_match_point.position_2d = a_pos
                new_trajectory.corr_image_points.append(new_match_point)

            trajectories.append(new_trajectory)
        # Save trajectories into database, use SFMCensusTrajectories table for temporary
        # if new_operation:
        #     sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id,
        #                                                       SFM_OPERATION.TRACK_USER_TAGGED_FEATURES,
        #                                                       list(range(self._start_image_id - 2, self._end_image_id)))
        #
        # self._db_ops.save_user_tagged_trajectories(self.session_id, sfm_operation_id, trajectories)

        return trajectories, of_error_set
