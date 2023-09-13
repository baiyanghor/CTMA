import os.path
import cv2 as cv
import math
import numpy as np
from typing import List
from scipy.stats import relfreq
from scipy.spatial import KDTree
from SFMOperations.SFMDataTypes import SFMImage, TowViewMatchedPoints, MatchedImagePoint, SFM_STATUS, SFMTrajectory,\
    BA_PHASE, SFM_OPERATION
from SFMOperations.SFMUtilities import get_ordered_image_ids_selection, get_shifted_range, rt_to_scene_from_vec,\
    rt_matrix_for_ba
from FileOperations.ImageFileOperators import ImageFileOperators
from SFMConfigure.SFMConfigureOperators import SFMConfigureOps
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.OpenCVOperators import OpenCVOperators
from SFMOperations.SFMBundleAdjustment import BundleAdjustment
from SFMOperations.TwoViewCalculator import TwoViewCalculator
from SFMOperations.MayaDataInterface import MayaDataInterface
from SFMOperations.VGGFunctions import X_from_xP_nonlinear


class CalNodeBase(object):

    def __init__(self, node_name='', session_id: str = '', work_dir=''):
        self.node_name = node_name
        self.data_dirty = False
        self.previous_node_name = ''
        self.work_dir = work_dir
        self.session_id = session_id
        self.dependent_nodes = []

    def set_work_dir(self, work_dir=''):
        self.work_dir = work_dir

    def set_previous_node_name(self, node_name=''):
        self.previous_node_name = node_name

    def __str__(self):
        return self.node_name


class ImageMatchedFeatures(CalNodeBase):

    def __init__(self, session_id='', node_name='ImageMatchedFeatures', work_dir=''):
        super(ImageMatchedFeatures, self).__init__(node_name, session_id, work_dir)
        self.__image_filename_list = []
        self.__image_file_ops = ImageFileOperators(work_dir)
        self.__config_ops = SFMConfigureOps(work_dir)
        self.__db_ops = MongodbInterface(work_dir)
        self.__opencv_ops = OpenCVOperators()
        self.__focal_length = 1.0
        self.__image_size = ()
        self.__image_count = 0

    def match_indices(self, image_index, max_index):
        a = image_index - 1
        if a < 0:
            a = 0
        b = image_index
        if b == max_index:
            b = image_index - 1
        return a, b

    def set_image_list(self):
        self.__focal_length, self.__image_size, self.__image_count, images_path = self.__db_ops.get_images_info(self.session_id)
        self.__image_filename_list = self.__image_file_ops.getImageRawFileList(images_path)

    def calculate_matched_features(self):
        match_relation = []
        image_mat_list = self.__opencv_ops.construct_mat_sequence(self.__image_filename_list)
        for i in range(self.__image_count - 1):
            points_a, points_b = self.__opencv_ops.twoview_get_matched_features(image_mat_list[i],
                                                                                image_mat_list[i + 1],
                                                                                extractor_name='OF')
            assert len(points_a) == len(points_b)
            a_match_relation = TowViewMatchedPoints()
            a_match_relation.matched_id = i
            a_match_relation.image_id_a = i
            a_match_relation.image_id_b = i + 1
            a_match_relation.matches_a = []
            a_match_relation.matches_b = []

            for j, (point_a, point_b) in enumerate(zip(points_a, points_b)):
                new_matched_point_a = MatchedImagePoint()
                new_matched_point_a.point_id = j
                new_matched_point_a.sfm_image_id = i
                new_matched_point_a.world_point_id = -1
                new_matched_point_a.position_2d = point_a
                a_match_relation.matches_a.append(new_matched_point_a)

                new_matched_point_b = MatchedImagePoint()
                new_matched_point_b.point_id = j
                new_matched_point_b.sfm_image_id = i + 1
                new_matched_point_b.world_point_id = -1
                new_matched_point_b.position_2d = point_b
                a_match_relation.matches_b.append(new_matched_point_b)

            match_relation.append(a_match_relation)

        sfm_images = []

        for i in range(self.__image_count):
            new_sfm_image = SFMImage()
            new_sfm_image.image_id = i
            new_sfm_image.pre_image_id = i - 1
            new_sfm_image.next_image_id = i + 1
            new_sfm_image.image_file_name = self.__image_filename_list[i]
            new_sfm_image.image_size = self.__image_size
            new_sfm_image.matches_to_pre = []
            new_sfm_image.matches_to_next = []

            matches_pre_idx, matches_next_idx = self.match_indices(i, self.__image_count - 1)

            new_sfm_image.matches_to_pre = match_relation[matches_pre_idx].matches_b
            new_sfm_image.matches_to_next = match_relation[matches_next_idx].matches_a

            sfm_images.append(new_sfm_image)

        # Deal with loop boundaries
        sfm_images[0].matches_to_pre = []
        sfm_images[-1].matches_to_next = []

        save_success = self.__db_ops.save_sfm_matched_features(self.session_id, sfm_images)
        status_success = self.__db_ops.set_session_status(self.session_id, SFM_STATUS.MATCHED_FEATURE_SAVED)

        if save_success and status_success:
            return True
        else:
            return False


class SFMInitialPair(CalNodeBase):
    def __init__(self, session_id='', node_name='SFMInitialPair', work_dir=''):
        super(SFMInitialPair, self).__init__(node_name, session_id, work_dir)
        self.__config_ops = SFMConfigureOps(work_dir)
        self.__db_ops = MongodbInterface(work_dir)
        self.__focal_length = 1.0
        self.__image_size = ()
        self.__image_count = 0

    def calculate_sfm_initial_pair(self):
        self.__focal_length, self.__image_size, self.__image_count, _ = self.__db_ops.get_images_info(self.session_id)

        sfm_image_list = self.__db_ops.get_sfm_image_list(self.session_id)

        height = self.__image_size[0]
        width = self.__image_size[1]
        x_bins = 16
        y_bins = math.floor(x_bins * height / width)

        distribution_list = []
        for a_sfm_image in sfm_image_list:
            match_pre = a_sfm_image.dump_matches_to_pre()
            match_next = a_sfm_image.dump_matches_to_next()

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

        selected_image_idx = int(np.argmin(np.array(distribution_list)))

        sfm_initial_pair = tuple()

        if 0 < selected_image_idx < self.__image_count - 1:
            if distribution_list[selected_image_idx - 1] > distribution_list[selected_image_idx + 1]:
                sfm_initial_pair = (selected_image_idx, selected_image_idx+1)
            else:
                sfm_initial_pair = (selected_image_idx-1, selected_image_idx)

        elif selected_image_idx == 0:
            sfm_initial_pair = (selected_image_idx, selected_image_idx + 1)

        elif selected_image_idx == self.__image_count - 1:
            sfm_initial_pair = (selected_image_idx - 1, selected_image_idx)

        self.__db_ops.save_sfm_initial_image_pair_id(self.session_id, sfm_initial_pair)
        self.__db_ops.set_session_status(self.session_id, SFM_STATUS.INITIAL_SFM_PAIR)

        return sfm_initial_pair


class InitialPNPExtrinsic(CalNodeBase):
    def __init__(self, session_id='', node_name='InitialPNPExtrinsic', work_dir=''):
        super(InitialPNPExtrinsic, self).__init__(node_name, session_id, work_dir)
        self.__db_ops = MongodbInterface(work_dir)

    def calculate_initial_pnp(self, bundle_adjustment=False):
        focal_length, image_size, image_count, images_path = self.__db_ops.get_images_info(self.session_id)

        world_points = self.__db_ops.get_user_world_points(self.session_id)
        image_points = self.__db_ops.get_user_images_points(self.session_id)
        sfm_initial_pair_ids = self.__db_ops.get_sfm_initial_pair_id(self.session_id)

        k = np.array([[focal_length, 0,            image_size[1]/2],
                      [0,            focal_length, image_size[0]/2],
                      [0,            0,            1              ]])

        ret_one, rvec_one, tvec_one = cv.solvePnP(world_points[4:], image_points[0, 4:], k, None,
                                                  flags=cv.SOLVEPNP_IPPE)
        ret_two, rvec_two, tvec_two = cv.solvePnP(world_points[4:], image_points[1, 4:], k, None,
                                                  flags=cv.SOLVEPNP_IPPE)

        ret_one, rvec_one, tvec_one, inliers_one = cv.solvePnPRansac(world_points, image_points[0],
                                                                     k, None, rvec_one, tvec_one,
                                                                     useExtrinsicGuess=True, iterationsCount=1000)
        ret_two, rvec_two, tvec_two, inliers_two = cv.solvePnPRansac(world_points, image_points[1],
                                                                     k, None, rvec_two, tvec_two,
                                                                     useExtrinsicGuess=True, iterationsCount=1000)

        if bundle_adjustment:
            camera_Rts = np.empty((2, 6), dtype=float)
            camera_Rts[0] = np.hstack([rvec_one.flatten(), tvec_one.flatten()])
            camera_Rts[1] = np.hstack([rvec_two.flatten(), tvec_two.flatten()])

            raw_shape = image_points.shape
            image_points_reshaped = image_points.reshape(raw_shape[0] * raw_shape[1], raw_shape[2])

            ba_ops = BundleAdjustment()
            residual_weight_threshold = 0.292
            ba_camera_Rts_weighted = ba_ops.ba_on_camera_paramter_weighted(camera_Rts, world_points,
                                                                           image_points_reshaped, k,
                                                                           residual_weight_threshold)
            cv_ops = OpenCVOperators()
            Rt_to_scene_one = cv_ops.cv_rt_to_matrix(ba_camera_Rts_weighted[0, :3], ba_camera_Rts_weighted[0, 3:])
            Rt_to_scene_two = cv_ops.cv_rt_to_matrix(ba_camera_Rts_weighted[1, :3], ba_camera_Rts_weighted[1, 3:])


        else:
            R_mat_one = cv.Rodrigues(rvec_one)[0]
            R_mat_two = cv.Rodrigues(rvec_two)[0]
            Rt_to_scene_one = np.hstack([R_mat_one, tvec_one])
            Rt_to_scene_two = np.hstack([R_mat_two, tvec_two])

        self.__db_ops.set_camera_rt_to_scene(self.session_id, sfm_initial_pair_ids[0], Rt_to_scene_one)
        self.__db_ops.set_camera_rt_to_scene(self.session_id, sfm_initial_pair_ids[1], Rt_to_scene_two)

        return True


class UserTaggedPNPExtrinsic(CalNodeBase):
    def __init__(self, node_name='UserTaggedPNPExtrinsic', session_id='', work_dir=''):
        super(UserTaggedPNPExtrinsic, self).__init__(node_name, session_id, work_dir)
        self._db_ops = MongodbInterface(work_dir)

    def calculate_tagged_pnp(self,
                             use_operation_ids: list,
                             new_sfm_operation=False,
                             sfm_operation_id='',
                             bundle_adjustment=False):
        if use_operation_ids is None or len(use_operation_ids) == 0:
            raise ValueError("Provide user tagged operation ID is required!")

        focal_length, image_size, _, _ = self._db_ops.get_images_info(self.session_id)
        user_tagged_op_id = use_operation_ids[0]
        user_tagged_frames, image_points, world_points =\
            self._db_ops.get_user_tagged_features(session_id=self.session_id, sfm_operation_id=user_tagged_op_id)

        k = np.array([[focal_length, 0,            image_size[1]/2],
                      [0,            focal_length, image_size[0]/2],
                      [0,            0,            1              ]])

        _, rvec_one, tvec_one = cv.solvePnP(world_points[4:], image_points[0, 4:], k, None, flags=cv.SOLVEPNP_IPPE)
        _, rvec_two, tvec_two = cv.solvePnP(world_points[4:], image_points[1, 4:], k, None, flags=cv.SOLVEPNP_IPPE)

        _, rvec_one, tvec_one, _ = cv.solvePnPRansac(world_points, image_points[0], k, None, rvec_one, tvec_one,
                                                     useExtrinsicGuess=True, iterationsCount=1000)
        _, rvec_two, tvec_two, _ = cv.solvePnPRansac(world_points, image_points[1], k, None, rvec_two, tvec_two,
                                                     useExtrinsicGuess=True, iterationsCount=1000)

        if bundle_adjustment:
            camera_rts = np.empty((2, 6), dtype=float)
            camera_rts[0] = np.hstack([rvec_one.flatten(), tvec_one.flatten()])
            camera_rts[1] = np.hstack([rvec_two.flatten(), tvec_two.flatten()])

            raw_shape = image_points.shape
            image_points_reshaped = image_points.reshape(raw_shape[0] * raw_shape[1], raw_shape[2])

            ba_ops = BundleAdjustment()
            residual_weight_threshold = 0.292
            ba_camera_rts_weighted = ba_ops.ba_on_camera_paramter_weighted(camera_rts, world_points,
                                                                           image_points_reshaped, k,
                                                                           residual_weight_threshold)
            cv_ops = OpenCVOperators()
            rt_to_scene_one = cv_ops.cv_rt_to_matrix(ba_camera_rts_weighted[0, :3], ba_camera_rts_weighted[0, 3:])
            rt_to_scene_two = cv_ops.cv_rt_to_matrix(ba_camera_rts_weighted[1, :3], ba_camera_rts_weighted[1, 3:])
        else:
            r_mat_one = cv.Rodrigues(rvec_one)[0]
            r_mat_two = cv.Rodrigues(rvec_two)[0]
            rt_to_scene_one = np.hstack([r_mat_one, tvec_one])
            rt_to_scene_two = np.hstack([r_mat_two, tvec_two])

        image_id_range = [frame - 1 for frame in user_tagged_frames]

        if new_sfm_operation:

            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id,
                                                              SFM_OPERATION.PNP_USER_TAGGED_FRAMES,
                                                              image_id_range)

        self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, image_id_range[0], rt_to_scene_one)
        self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, image_id_range[1], rt_to_scene_two)

        return True


class CensusTrajectoryPairs(CalNodeBase):
    def __init__(self, session_id='', node_name='CensusTrajectories', work_dir=''):
        super(CensusTrajectoryPairs, self).__init__(node_name, session_id, work_dir)
        self.__db_ops = MongodbInterface(work_dir)

    def calculate_trajectories(self, pix_tolerance=0.0):
        sfm_image_list = self.__db_ops.get_sfm_image_list(self.session_id)
        _, _, image_count, _ = self.__db_ops.get_images_info(self.session_id)
        for sfm_image_id in range(1, image_count - 1):
            current_sfm_image = sfm_image_list[sfm_image_id]
            new_trajectory_pair = list()
            matched_next_list = current_sfm_image.dump_matches_to_next()
            matched_pre_list = current_sfm_image.dump_matches_to_pre()

            if len(matched_pre_list) > 0 and len(matched_next_list) > 0:
                searcher = KDTree(matched_pre_list)
                search_radius = pix_tolerance
                trajectory_indices_on_previous = searcher.query_ball_point(matched_next_list,
                                                                           r=search_radius)
                for next_id, indices in enumerate(trajectory_indices_on_previous):
                    if len(indices) == 1:
                        new_trajectory_pair.append([indices[0], next_id])
                    elif len(indices) > 1:
                        local_searcher = KDTree(matched_pre_list[indices])
                        local_indices = local_searcher.query(matched_next_list[next_id])
                        new_trajectory_pair.append([indices[0] + local_indices[0], next_id])

            if len(new_trajectory_pair) > 0:
                current_sfm_image.trajectory_pair = np.array(new_trajectory_pair)
            else:
                current_sfm_image.trajectory_pair = None

        # Deal with first and last frame
        if sfm_image_list[1].trajectory_pair is None:
            sfm_image_list[0].trajectory_pair = None
        else:
            minus_ones = -1 * np.ones((sfm_image_list[1].trajectory_pair.shape[0], 1), dtype=np.int)
            sfm_image_list[0].trajectory_pair = \
                np.hstack([minus_ones, sfm_image_list[1].trajectory_pair[:, [0]]])

        if sfm_image_list[-2].trajectory_pair is None:
            sfm_image_list[-1].trajectory_pair = None
        else:
            minus_ones = -1 * np.ones((sfm_image_list[-2].trajectory_pair.shape[0], 1), dtype=np.int)
            sfm_image_list[-1].trajectory_pair = \
                np.hstack([sfm_image_list[-2].trajectory_pair[:, [1]], minus_ones])

        self.__db_ops.set_sfm_image_trajectory_pair(self.session_id, sfm_image_list)
        self.__db_ops.set_session_status(self.session_id, SFM_STATUS.TRAJECTORY_PAIRS_READY)

        return True


class MaskingFeatures(CalNodeBase):
    def __init__(self, session_id='', node_name='MaskingFeatures', work_dir=''):
        super(MaskingFeatures, self).__init__(node_name, session_id, work_dir)
        self.__image_file_ops = ImageFileOperators(work_dir)
        self.__conf_ops = SFMConfigureOps(work_dir)
        self.__cv_ops = OpenCVOperators()
        self.__db_ops = MongodbInterface(work_dir)

    def generate_mask(self):
        _, _, _, images_path = self.__db_ops.get_images_info(self.session_id)
        image_filename_list = self.__image_file_ops.getImageRawFileList(images_path)

        shadow_mask_path = os.path.join(self.work_dir, self.__conf_ops.get_shadow_mask_path())
        shadow_mask_path = os.path.join(shadow_mask_path, self.session_id)

        mat_list = self.__cv_ops.construct_mat_sequence(image_filename_list)
        mask_image_files = self.__cv_ops.generate_shadow_mask_list(shadow_mask_path, mat_list)

        for i, mask_file in enumerate(mask_image_files):
            self.__db_ops.set_mask_file_name(session_id=self.session_id, image_id=i, mask_file_name=mask_file)

        return mask_image_files

    def masking_features(self, refresh_mask=False):
        if refresh_mask:
            self.generate_mask()

        sfm_image_list = self.__db_ops.get_sfm_image_list(self.session_id)

        for a_sfm_image in sfm_image_list:
            mask_mat = cv.imread(a_sfm_image.mask_file_name, flags=cv.IMREAD_GRAYSCALE)
            features_to_pre = a_sfm_image.dump_matches_to_pre().astype("int")
            features_to_next = a_sfm_image.dump_matches_to_next().astype("int")

            if features_to_pre.shape[0] > 0:
                mask_values_to_pre = mask_mat[features_to_pre[:, 1], features_to_pre[:, 0]]
                keep_indices_to_pre = np.where(mask_values_to_pre == 0)[0]
            else:
                keep_indices_to_pre = None

            if features_to_next.shape[0] > 0:
                mask_values_to_next = mask_mat[features_to_next[:, 1], features_to_next[:, 0]]
                keep_indices_to_next = np.where(mask_values_to_next == 0)[0]
            else:
                keep_indices_to_next = None

            self.__db_ops.set_masked_feature_indices(self.session_id, a_sfm_image.image_id,
                                                     keep_indices_to_pre, keep_indices_to_next)

        return True


class TwoViewSFMForward(CalNodeBase):
    def __init__(self, session_id='', node_name='TwoViewSFMForward', work_dir='', start_image_id=-1, end_image_id=-1):
        super(TwoViewSFMForward, self).__init__(node_name, session_id, work_dir)
        self.__start_image_id = start_image_id
        self.__end_image_id = end_image_id
        self.__db_ops = MongodbInterface(work_dir)
        self.__twoview_calculator = TwoViewCalculator()

    def twoview_sfm_forward_v2(self, with_trajectory_filter=True, with_mask_filter=False, bundle_adjust=False,
                            adjust_window=3, weight_threshold=0.292):

        def get_incremental_trajectory_section(traj_list: List[SFMTrajectory], start_id: int):
            start_list = []
            adjacent_list = []
            for a_traj in traj_list:
                if start_id + 1 < a_traj.start_image_id + a_traj.traject_length:
                    for i, matched_point in enumerate(a_traj.corr_image_points):
                        if matched_point.sfm_image_id == start_id and\
                                a_traj.corr_image_points[i + 1].sfm_image_id == start_id + 1:
                            start_list.append(matched_point.position_2d)
                            adjacent_list.append(a_traj.corr_image_points[i + 1].position_2d)
                            break

            return start_list, adjacent_list

        assert self.__start_image_id < self.__end_image_id

        # trajectory_pairs = self.__db_ops.get_trajectory_pairs(self.session_id)
        # sfm_image_list = self.__db_ops.get_sfm_image_list(self.session_id)
        trajectory_list = self.__db_ops.get_forward_trajectory_list(self.session_id, self.__start_image_id)

        focal_length, image_size, image_count, image_path = self.__db_ops.get_images_info(self.session_id)
        maya_data_ops = MayaDataInterface(self.session_id, self.work_dir)

        for idx in range(self.__start_image_id, self.__end_image_id):
            start_point_list, adjacent_point_list = get_incremental_trajectory_section(trajectory_list, idx)
            if len(start_point_list) + len(adjacent_point_list) == 0:
                print("Trajectory is not enough")
                return

            start_select_match_to_next = np.array(start_point_list)
            next_select_match_to_pre = np.array(adjacent_point_list)
            #
            self.__twoview_calculator.set_focal_length(focal_length)
            self.__twoview_calculator.set_image_size(image_size)
            self.__twoview_calculator.initial_camera_intrinsic()
            self.__twoview_calculator.set_correspondences(start_select_match_to_next, next_select_match_to_pre)
            # # relative_camera_pose = self.__twoview_calculator.cv_get_relative_camera_pose()
            relative_camera_pose = self.__twoview_calculator.calc_relative_camera_pose()
            start_Rt_to_scene = np.array(self.__db_ops.get_camera_rt_to_scene(self.session_id, idx))
            #
            relative_camera_pose_trans = np.vstack([relative_camera_pose, np.array([[0, 0, 0, 1]])])
            start_Rt_to_scene_trans = np.vstack([start_Rt_to_scene, np.array([[0, 0, 0, 1]])])
            next_Rt_to_scene_trans = np.matmul(relative_camera_pose_trans, start_Rt_to_scene_trans)
            next_Rt_to_scene = next_Rt_to_scene_trans[:3, :]
            #
            #
            # if bundle_adjust:
            #     ba_start = max(self.__start_image_id, idx - adjust_window)
            #     camera_Rts = np.empty((idx - ba_start, 6))
            #     camera_matrix = self.__twoview_calculator.non_center_camera_matrix()
            #
            #     for ba_idx in range(ba_start, idx + 1):
            #         previouse_Rt_to_scene = np.array(self.__db_ops.get_camera_Rt_to_scene(self.session_id, ba_idx))
            #
            #         start_pm = np.matmul(camera_matrix, start_Rt_to_scene)
            #         next_pm = np.matmul(camera_matrix, next_Rt_to_scene)
            #         world_points = self.__twoview_calculator.triangulate_two_view(start_pm, next_pm,
            #                                                                       start_select_match_to_next,
            #                                                                       next_select_match_to_pre,
            #                                                                       image_size)
            #         world_point_3d = world_points.squeeze(axis=2)[:, :3]
            #         start_rvec = cv.Rodrigues(start_Rt_to_scene[:, :3])[0].flatten()
            #         start_tvec = start_Rt_to_scene[:, [3]].flatten()
            #
            #     next_rvec = cv.Rodrigues(next_Rt_to_scene[:, :3])[0].flatten()
            #     next_tvec = next_Rt_to_scene[:, [3]].flatten()
            #     camera_Rts = np.vstack([np.hstack([start_rvec, start_tvec]), np.hstack([next_rvec, next_tvec])])
            #     image_pts = np.vstack([start_select_match_to_next, next_select_match_to_pre])
            #
            #     ba_ops = BundleAdjustment()
            #     cam_rvecs, cam_tvecs, ba_world_pts = \
            #         ba_ops.two_view_ba_with_sparsity_matrix_wieght_update(camera_Rts,
            #                                                               world_point_3d,
            #                                                               image_pts,
            #                                                               camera_matrix,
            #                                                               weight_threshold)
            #     start_r_mat = cv.Rodrigues(cam_rvecs[0])[0]
            #     next_r_mat = cv.Rodrigues(cam_rvecs[1])[0]
            #     result_start_Rt_to_scene = np.hstack([start_r_mat, cam_tvecs[0].reshape(3, 1)])
            #     result_next_Rt_to_scene = np.hstack([next_r_mat, cam_tvecs[1].reshape(3, 1)])
            #     # self.__db_ops.set_camera_Rt_to_scene(self.session_id, idx, result_start_Rt_to_scene)
            #     self.__db_ops.set_camera_Rt_to_scene(self.session_id, idx + 1, result_next_Rt_to_scene)
            #
            # else:
            #     self.__db_ops.set_camera_Rt_to_scene(self.session_id, idx + 1, next_Rt_to_scene)
            #     # self.__db_ops.set_camera_Rt_to_scene(self.session_id, idx + 1, relative_camera_pose)

        return True

    def twoview_sfm_forward(self, with_trajectory_filter=True, with_mask_filter=False, bundle_adjust=False,
                            weight_threshold=0.292):
        trajectory_pairs = self.__db_ops.get_trajectory_pairs(self.session_id)
        sfm_image_list = self.__db_ops.get_sfm_image_list(self.session_id)
        assert len(trajectory_pairs) == len(sfm_image_list) and len(sfm_image_list) > 0

        focal_length, image_size, image_count, image_path = self.__db_ops.get_images_info(self.session_id)
        maya_data_ops = MayaDataInterface(self.session_id, self.work_dir)
        if self.__start_image_id < self.__end_image_id:
            for idx in range(self.__start_image_id, self.__end_image_id):
                start_sfm_image = sfm_image_list[idx]
                next_sfm_image = sfm_image_list[idx+1]
                start_trajectory_pair = np.array(trajectory_pairs[idx])
                next_trajectory_pair = np.array(trajectory_pairs[idx+1])
                start_match_to_next = start_sfm_image.dump_matches_to_next()
                next_match_to_pre = next_sfm_image.dump_matches_to_pre()

                if with_trajectory_filter:
                    intersect_idx = np.intersect1d(start_trajectory_pair[:, 1],
                                                   next_trajectory_pair[:, 0]).astype('int')
                    start_select_match_to_next = start_match_to_next[intersect_idx]
                    next_select_match_to_pre = next_match_to_pre[intersect_idx]
                else:
                    start_select_match_to_next = start_match_to_next
                    next_select_match_to_pre = next_match_to_pre

                self.__twoview_calculator.set_focal_length(focal_length)
                self.__twoview_calculator.set_image_size(image_size)
                self.__twoview_calculator.initial_camera_intrinsic()
                self.__twoview_calculator.set_correspondences(start_select_match_to_next, next_select_match_to_pre)
                # relative_camera_pose = self.__twoview_calculator.cv_get_relative_camera_pose()
                relative_camera_pose = self.__twoview_calculator.calc_relative_camera_pose()
                start_Rt_to_scene = np.array(self.__db_ops.get_camera_rt_to_scene(self.session_id, idx))

                relative_camera_pose_trans = np.vstack([relative_camera_pose, np.array([[0, 0, 0, 1]])])
                start_Rt_to_scene_trans = np.vstack([start_Rt_to_scene, np.array([[0, 0, 0, 1]])])
                next_Rt_to_scene_trans = np.matmul(relative_camera_pose_trans, start_Rt_to_scene_trans)
                next_Rt_to_scene = next_Rt_to_scene_trans[:3, :]

                if bundle_adjust:
                    camera_matrix = self.__twoview_calculator.non_center_camera_matrix()
                    start_pm = np.matmul(camera_matrix, start_Rt_to_scene)
                    next_pm = np.matmul(camera_matrix, next_Rt_to_scene)
                    world_points = self.__twoview_calculator.triangulate_two_view(start_pm, next_pm,
                                                                                  start_select_match_to_next,
                                                                                  next_select_match_to_pre,
                                                                                  image_size)
                    world_point_3d = world_points.squeeze(axis=2)[:, :3]
                    start_rvec = cv.Rodrigues(start_Rt_to_scene[:, :3])[0].flatten()
                    start_tvec = start_Rt_to_scene[:, [3]].flatten()
                    next_rvec = cv.Rodrigues(next_Rt_to_scene[:, :3])[0].flatten()
                    next_tvec = next_Rt_to_scene[:, [3]].flatten()
                    camera_Rts = np.vstack([np.hstack([start_rvec, start_tvec]), np.hstack([next_rvec, next_tvec])])
                    image_pts = np.vstack([start_select_match_to_next, next_select_match_to_pre])

                    ba_ops = BundleAdjustment()
                    cam_rvecs, cam_tvecs, ba_world_pts = \
                        ba_ops.two_view_ba_with_sparsity_matrix_wieght_update(camera_Rts,
                                                                              world_point_3d,
                                                                              image_pts,
                                                                              camera_matrix,
                                                                              weight_threshold)
                    start_r_mat = cv.Rodrigues(cam_rvecs[0])[0]
                    next_r_mat = cv.Rodrigues(cam_rvecs[1])[0]
                    result_start_Rt_to_scene = np.hstack([start_r_mat, cam_tvecs[0].reshape(3, 1)])
                    result_next_Rt_to_scene = np.hstack([next_r_mat, cam_tvecs[1].reshape(3, 1)])
                    # self.__db_ops.set_camera_Rt_to_scene(self.session_id, idx, result_start_Rt_to_scene)
                    self.__db_ops.set_camera_rt_to_scene(self.session_id, idx + 1, result_next_Rt_to_scene)

                else:
                    self.__db_ops.set_camera_rt_to_scene(self.session_id, idx + 1, next_Rt_to_scene)
                    # self.__db_ops.set_camera_Rt_to_scene(self.session_id, idx + 1, relative_camera_pose)

        return True


class NViewSFMForward(CalNodeBase):
    def __init__(self, session_id='', node_name='NViewSFMForward', work_dir='', start_image_id=-1,
                 frame_window=1,
                 end_image_id=-1):

        super(NViewSFMForward, self).__init__(node_name, session_id, work_dir)
        self._start_image_id = start_image_id
        self._frame_window = frame_window
        self._end_image_id = end_image_id
        self._db_ops = MongodbInterface(self.work_dir)
        self.dependent_nodes = ['TrajectoryForward', 'UserTaggedPNPExtrinsic']

    def calculate_sfm_forward(self, bundle_adjust=False, weight_threshold=0.292):
        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        # Assume the position of self.__start_image_id prior known
        trajectory_list = self._db_ops.get_forward_trajectory_list(self.session_id, self._start_image_id, self._frame_window)
        # Construct world point list and image list
        point_count_each = len(trajectory_list)
        image_points_list = np.empty((self._frame_window, point_count_each, 2))
        for i in range(point_count_each):
            for j in range(self._frame_window):
                image_points_list[j, i] = trajectory_list[i].corr_image_points[j].position_2d

        prestart_Rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id - 1))
        start_Rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id))
        start_trans_scene = np.vstack([start_Rt_to_scene, np.array([[0, 0, 0, 1]])])
        maya_data_ops = MayaDataInterface()
        rotate_pre, camera_c_pre = maya_data_ops.camera_rt_to_maya(prestart_Rt_to_scene)
        rotate_start, camera_c_start = maya_data_ops.camera_rt_to_maya(start_Rt_to_scene)
        last_cam_distance = np.linalg.norm(camera_c_start - camera_c_pre)

        start_R = start_Rt_to_scene[:, :3]
        start_t = start_Rt_to_scene[:, [3]]
        rvec = cv.Rodrigues(start_R)[0].flatten()
        tvec = start_t.flatten()
        Rts = np.zeros((self._frame_window, 6))
        Rts[0] = np.hstack([rvec, tvec])
        Rt_matices = np.zeros((self._frame_window, 3, 4))
        Rt_matices[0] = start_Rt_to_scene

        twoview_ops = TwoViewCalculator()
        twoview_ops.set_focal_length(focal_length)
        twoview_ops.set_image_size(image_size)
        twoview_ops.initial_camera_intrinsic()
        twoview_ops.non_center_camera_matrix()

        for idx in range(1, self._frame_window):
            twoview_ops.set_correspondences(image_points_list[idx-1], image_points_list[idx])
            normal_rel_Rt = twoview_ops.calc_relative_camera_pose()
            # normal_rel_Rt = twoview_ops.cv_get_relative_camera_pose()
            estimate_rel_Rt = np.hstack([normal_rel_Rt[:, :3], last_cam_distance * normal_rel_Rt[:, [3]]])
            relative_trans_matrix = np.vstack([estimate_rel_Rt, np.array([[0, 0, 0, 1]])])
            estimate_trans_matrix = np.matmul(relative_trans_matrix, start_trans_scene)
            Rt_matices[idx] = estimate_trans_matrix[:3, :]
            estimate_R = estimate_trans_matrix[:3, :3]
            estimate_t = estimate_trans_matrix[:3, [3]]
            rvec = cv.Rodrigues(estimate_R)[0].flatten()
            tvec = estimate_t.flatten()
            Rts[idx] = np.hstack([rvec, tvec])
            start_trans_scene = estimate_trans_matrix.copy()

        if bundle_adjust:
            homo_world_points = np.zeros((point_count_each, 4, 1))
            # Triangulate
            camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                      [0, focal_length, image_size[0] / 2.0],
                                      [0, 0, 1]])

            project_matrices = np.matmul(camera_matrix, Rt_matices)
            image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
            for i in range(point_count_each):
                homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)

            world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)

            ba_ops = BundleAdjustment()
            image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
            cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(Rts, world_pts, image_point,
                                                                                            camera_matrix,
                                                                                            weight_threshold)
            for i in range(1, self._frame_window):
                cam_r_mat = cv.Rodrigues(cam_rveces[i])[0]
                Rt_to_scene = np.hstack([cam_r_mat, cam_tveces[i].reshape(3, 1)])
                self._db_ops.set_camera_rt_to_scene(self.session_id, self._start_image_id + i, Rt_to_scene)
        else:
            for i in range(1, self._frame_window):
                self._db_ops.set_camera_rt_to_scene(self.session_id, self._start_image_id + i, Rt_matices[i])

    def calculate_sfm_forward_retrajectory(self, bundle_adjust=False, weight_threshold=0.292, min_trajectories=8):

        def is_rest_traj_enough(traj_list: List[SFMTrajectory],
                                start_id: int,
                                window_size: int,
                                least_rest_traj_count=8):
            keep_indices = []
            count = 0
            for i, a_traj in enumerate(traj_list):
                if a_traj.traject_length - (start_id - a_traj.start_image_id) > window_size:
                    keep_indices.append(i)
                    count += 1

            if count < least_rest_traj_count:
                print(f"---------------- Not enough from {start_id} count {count} lower than {least_rest_traj_count}"
                      f"\n---------------- Trajectory start from {traj_list[0].start_image_id} {traj_list[1].start_image_id}")
                return False, []

            return True, keep_indices

        def refresh_trajectory_list(session_id: str, work_dir: str, start_image_id: int, end_image_id: int,
                                    frame_window: int):
            traj_forward_ops = TrajectoryForward(session_id=session_id, work_dir=work_dir,
                                                 start_image_id=start_image_id, end_image_id=end_image_id)

            traj_forward_ops.calculate_trajectory_forward()
            new_trajectory_list = self._db_ops.get_forward_trajectory_list(self.session_id,
                                                                           start_image_id,
                                                                           frame_window)
            return new_trajectory_list

        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        # Check if trajectories is enough or recalculate trajectories from current image id
        trajectory_list = refresh_trajectory_list(self.session_id,
                                                  self.work_dir,
                                                  self._start_image_id,
                                                  self._end_image_id,
                                                  self._frame_window)

        # Construct world point list and image list
        point_count_each = len(trajectory_list)
        if point_count_each < min_trajectories:
            print(f"==== Not enough trajectories from start")
            return

        prestart_Rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id - 1))
        start_Rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id))
        maya_data_ops = MayaDataInterface()
        rotate_pre, camera_c_pre = maya_data_ops.camera_rt_to_maya(prestart_Rt_to_scene)
        rotate_start, camera_c_start = maya_data_ops.camera_rt_to_maya(start_Rt_to_scene)
        assume_cam_distance = np.linalg.norm(camera_c_start - camera_c_pre)

        twoview_ops = TwoViewCalculator()
        twoview_ops.set_focal_length(focal_length)
        twoview_ops.set_image_size(image_size)
        twoview_ops.initial_camera_intrinsic()
        twoview_ops.non_center_camera_matrix()

        for start_idx in range(self._start_image_id, self._end_image_id - self._frame_window + 2):

            print(f"<<<<<<<<<<<<<<<<< Dealing with image {start_idx}"
                  f"\n<<<<<<<<<<<<<<<<< Trajectory count {len(trajectory_list)}")

            enough, keep_indices = is_rest_traj_enough(traj_list=trajectory_list,
                                                       start_id=start_idx,
                                                       window_size=self._frame_window,
                                                       least_rest_traj_count=min_trajectories)
            if not enough:
                print(f"Not enough when refresh trajectories at image_id {start_idx}")
                return

            rest_trajectory_list = []
            for idx in keep_indices:
                rest_trajectory_list.append(trajectory_list[idx])

            point_count_each = len(rest_trajectory_list)
            if point_count_each < min_trajectories:
                print(f"====Not enough trajectories from {start_idx}")
                return

            image_points_list = np.empty((self._frame_window, point_count_each, 2))
            for i in range(point_count_each):
                offset = start_idx - rest_trajectory_list[i].start_image_id
                for j in range(self._frame_window):
                    image_points_list[j, i] = rest_trajectory_list[i].corr_image_points[j + offset].position_2d

            print(f">>>>>>>>>>>>>>>>>>> Dealing with image {start_idx}"
                  f"\n>>>>>>>>>>>>>>>>>>> Trajectory count {point_count_each}")

            start_Rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, start_idx))
            start_trans_scene = np.vstack([start_Rt_to_scene, np.array([[0, 0, 0, 1]])])

            start_R = start_Rt_to_scene[:, :3]
            start_t = start_Rt_to_scene[:, [3]]
            rvec = cv.Rodrigues(start_R)[0].flatten()
            tvec = start_t.flatten()
            Rts = np.zeros((self._frame_window, 6))
            Rts[0] = np.hstack([rvec, tvec])
            Rt_matices = np.zeros((self._frame_window, 3, 4))
            Rt_matices[0] = start_Rt_to_scene

            for idx in range(1, self._frame_window):
                twoview_ops.set_correspondences(image_points_list[idx-1], image_points_list[idx])
                normal_rel_Rt = twoview_ops.calc_relative_camera_pose()
                # normal_rel_Rt = twoview_ops.cv_get_relative_camera_pose()
                estimate_rel_Rt = np.hstack([normal_rel_Rt[:, :3], assume_cam_distance * normal_rel_Rt[:, [3]]])
                relative_trans_matrix = np.vstack([estimate_rel_Rt, np.array([[0, 0, 0, 1]])])
                estimate_trans_matrix = np.matmul(relative_trans_matrix, start_trans_scene)
                Rt_matices[idx] = estimate_trans_matrix[:3, :]
                estimate_R = estimate_trans_matrix[:3, :3]
                estimate_t = estimate_trans_matrix[:3, [3]]
                rvec = cv.Rodrigues(estimate_R)[0].flatten()
                tvec = estimate_t.flatten()
                Rts[idx] = np.hstack([rvec, tvec])
                start_trans_scene = estimate_trans_matrix.copy()

            if bundle_adjust:
                homo_world_points = np.zeros((point_count_each, 4, 1))
                # Triangulate
                camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                          [0, focal_length, image_size[0] / 2.0],
                                          [0, 0, 1]])

                project_matrices = np.matmul(camera_matrix, Rt_matices)
                image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
                for i in range(point_count_each):
                    homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)

                world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)

                ba_ops = BundleAdjustment()
                image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
                cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(Rts, world_pts,
                                                                                                image_point,
                                                                                                camera_matrix,
                                                                                                weight_threshold)
                for i in range(1, self._frame_window):
                    cam_r_mat = cv.Rodrigues(cam_rveces[i])[0]
                    Rt_to_scene = np.hstack([cam_r_mat, cam_tveces[i].reshape(3, 1)])
                    self._db_ops.set_camera_rt_to_scene(self.session_id, start_idx + i, Rt_to_scene)
                    print(f"IIIIIIIIIIIIIIIIIIIIIIIIIII Image {start_idx + i} saved")
            else:
                for i in range(1, self._frame_window):
                    self._db_ops.set_camera_rt_to_scene(self.session_id, start_idx + i, Rt_matices[i])

            # Check next frame has enough trajectories or not
            enough, keep_indices = is_rest_traj_enough(traj_list=rest_trajectory_list,
                                                       start_id=start_idx + 1,
                                                       window_size=self._frame_window,
                                                       least_rest_traj_count=min_trajectories)

            if not enough:
                recalculate_start = start_idx
                trajectory_list = refresh_trajectory_list(self.session_id, self.work_dir, recalculate_start,
                                                          self._end_image_id, self._frame_window)

                print(f"---------------- Refresh trajectories from {recalculate_start}")

    def calculate_sfm_forward_convolution_ba(self,
                                             bundle_adjust=False,
                                             weight_threshold=0.292,
                                             min_trajectories=8,
                                             conv_back=2):
        """
        Global self.__start_image_id start from image that not from PNP,
        For test two frame for PNP, so the set conv_back 2
        :param bundle_adjust:
        :param weight_threshold:
        :param min_trajectories:
        :param conv_back:
        :return:
        """
        def is_rest_traj_enough(traj_list: List[SFMTrajectory],
                                start_id: int,
                                window_size: int,
                                least_rest_traj_count=8):
            to_keep_indices = []
            count = 0
            for i, a_traj in enumerate(traj_list):
                if a_traj.traject_length > (start_id - a_traj.start_image_id) >= (window_size - 1):
                    to_keep_indices.append(i)
                    count += 1

            if count < least_rest_traj_count:
                print(f"---------------- Not enough from {start_id} count {count} lower than {least_rest_traj_count}"
                      f"\n---------------- Trajectory start from {traj_list[0].start_image_id} {traj_list[1].start_image_id}")
                return False, []

            return True, to_keep_indices

        def refresh_trajectory_list(session_id: str, work_dir: str, start_image_id: int, end_image_id: int,
                                    frame_window: int):
            traj_forward_ops = TrajectoryForward(session_id=session_id, work_dir=work_dir,
                                                 start_image_id=(start_image_id - frame_window + 1),
                                                 end_image_id=end_image_id)

            traj_forward_ops.calculate_trajectory_forward()
            new_trajectory_list = self._db_ops.get_forward_trajectory_list(self.session_id,
                                                                           (start_image_id - frame_window + 1),
                                                                           frame_window)
            return new_trajectory_list

        assert self._start_image_id + 1 > self._frame_window
        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        # Check if trajectories is enough or recalculate trajectories from current image id
        trajectory_list = refresh_trajectory_list(self.session_id,
                                                  self.work_dir,
                                                  self._start_image_id,
                                                  self._end_image_id,
                                                  self._frame_window)
        pnp_one_Rt_to_scene = np.array(
            self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id - self._frame_window + 1))
        pnp_two_Rt_to_scene = np.array(
            self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id - self._frame_window + 2))

        maya_data_ops = MayaDataInterface()
        rotate_one, camera_c_one = maya_data_ops.camera_rt_to_maya(pnp_one_Rt_to_scene)
        rotate_two, camera_c_two = maya_data_ops.camera_rt_to_maya(pnp_two_Rt_to_scene)
        assume_cam_distance = np.linalg.norm(camera_c_two - camera_c_one)
        # Maybe I can fix this but alter the way to calculate new camera Rt
        # assume_cam_distance = np.linalg.norm(pnp_two_Rt_to_scene[:, [3]].flatten() - pnp_one_Rt_to_scene[:, [3]].flatten())

        twoview_ops = TwoViewCalculator()
        twoview_ops.set_focal_length(focal_length)
        twoview_ops.set_image_size(image_size)
        twoview_ops.initial_camera_intrinsic()
        twoview_ops.non_center_camera_matrix()
        sfm_images = self._db_ops.get_sfm_image_list(self.session_id)
        for start_idx in range(self._start_image_id, self._end_image_id):

            print(f"<<<<<<<<<<<<<<<<< Dealing with image {start_idx}"
                  f"\n<<<<<<<<<<<<<<<<< Trajectory count {len(trajectory_list)}")

            enough, keep_indices = is_rest_traj_enough(traj_list=trajectory_list,
                                                       start_id=start_idx,
                                                       window_size=self._frame_window,
                                                       least_rest_traj_count=min_trajectories)
            if not enough:
                print(f"Not enough when refresh trajectories at image_id {start_idx}")
                return

            rest_trajectory_list = []
            for idx in keep_indices:
                rest_trajectory_list.append(trajectory_list[idx])

            point_count_each = len(rest_trajectory_list)
            if point_count_each < min_trajectories:
                print(f"====Not enough trajectories from {start_idx}")
                return

            image_points_list = np.empty((self._frame_window, point_count_each, 2))
            for i in range(point_count_each):
                offset = start_idx - rest_trajectory_list[i].start_image_id - self._frame_window + 1
                for j in range(self._frame_window):
                    print(f"CCCCCCCC Check offset {offset} on frame trajectory {i} start_image_id: {rest_trajectory_list[i].start_image_id}")
                    image_points_list[j, i] = rest_trajectory_list[i].corr_image_points[j + offset].position_2d

            print(f">>>>>>>>>>>>>>>>>>> Dealing with image {start_idx}"
                  f"\n>>>>>>>>>>>>>>>>>>> Trajectory count {point_count_each}")

            rt_for_ba_list = np.zeros((self._frame_window, 6))
            rt_matrices = np.zeros((self._frame_window, 3, 4))
            for i, image_idx in enumerate(range(start_idx - self._frame_window + 1, start_idx)):
                print(f"RRRRRRRRRRRRRRRRR Read perivous {image_idx}")
                rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, image_idx))
                rt_matrices[i] = rt_to_scene
                rt_for_ba_list[i] = rt_matrix_for_ba(rt_to_scene)

            last_rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, start_idx - 1))
            # Get relative camera pose
            print(f"CCCCCCCCCCCCC Check correspondences")
            print(image_points_list[self._frame_window - 2].shape)
            print(image_points_list[self._frame_window - 2])
            print("---------------------------------------")
            print(image_points_list[self._frame_window - 1].shape)
            print(image_points_list[self._frame_window - 1])
            print(f"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
            # This should use raw point correspondences
            # twoview_ops.set_correspondences(image_points_list[self.__frame_window - 2],
            #                                 image_points_list[self.__frame_window - 1])

            twoview_ops.set_correspondences(sfm_images[start_idx-1].dump_matches_to_next(),
                                            sfm_images[start_idx].dump_matches_to_pre())

            normal_rel_Rt = twoview_ops.calc_relative_camera_pose()

            estimate_rel_Rt = np.hstack([normal_rel_Rt[:, :3], assume_cam_distance * normal_rel_Rt[:, [3]]])
            relative_trans_matrix = np.vstack([estimate_rel_Rt, np.array([[0, 0, 0, 1]])])
            last_trans_matrix = np.vstack([last_rt_to_scene, np.array([0, 0, 0, 1])])
            estimate_trans_matrix = np.matmul(relative_trans_matrix, last_trans_matrix)
            rt_matrices[self._frame_window - 1] = estimate_trans_matrix[:3, :]
            rt_for_ba_list[self._frame_window - 1] = rt_matrix_for_ba(estimate_trans_matrix[:3, :])

            if bundle_adjust:
                homo_world_points = np.zeros((point_count_each, 4, 1))
                # Triangulate
                camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                          [0, focal_length, image_size[0] / 2.0],
                                          [0, 0, 1]])

                project_matrices = np.matmul(camera_matrix, rt_matrices)
                image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
                for i in range(point_count_each):
                    homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)

                world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)
                print("CCCCCCCCCCCCCC Check world points")
                print(world_pts.shape)
                print(world_pts)
                print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")

                ba_ops = BundleAdjustment()
                image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
                cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rt_for_ba_list,
                                                                                                world_pts, image_point,
                                                                                                camera_matrix,
                                                                                                weight_threshold)
                cam_r_mat = cv.Rodrigues(cam_rveces[-1])[0]
                Rt_to_scene = np.hstack([cam_r_mat, cam_tveces[-1].reshape(3, 1)])
                self._db_ops.set_camera_rt_to_scene(self.session_id, start_idx, Rt_to_scene)
                print(f"IIIIIIIIIIIIIIIIIIIIIIIIIII Image {start_idx} saved")
            else:
                self._db_ops.set_camera_rt_to_scene(self.session_id, start_idx, rt_matrices[-1])

            # Check next frame has enough trajectories or not
            # enough, keep_indices = is_rest_traj_enough(traj_list=trajectory_list,
            #                                            start_id=start_idx + 1,
            #                                            window_size=self.__frame_window,
            #                                            least_rest_traj_count=min_trajectories)
            #
            # if not enough:
            #     recalculate_start = start_idx
            #     trajectory_list = refresh_trajectory_list(self.session_id, self.work_dir, recalculate_start,
            #                                               self.__end_image_id, self.__frame_window)
            #
            #     print(f"---------------- Refresh trajectories from {recalculate_start}")

    def calculate_sfm_forward_incremental_ba(self,
                                             bundle_adjust=False,
                                             weight_threshold=0.292):
        """
        In this function self.__start_image_id means the first frame after PnP frames
        self.__frame_window means PnP frames ahead
        :param bundle_adjust:
        :param weight_threshold:
        :return:
        """
        def refresh_trajectory_list(session_id: str, work_dir: str, start_image_id: int, end_image_id: int,
                                    frame_window: int):
            traj_forward_ops = TrajectoryForward(session_id=session_id, work_dir=work_dir,
                                                 start_image_id=start_image_id, end_image_id=end_image_id)

            traj_forward_ops.calculate_trajectory_forward()
            new_trajectory_list = self._db_ops.get_forward_trajectory_list(self.session_id,
                                                                           start_image_id,
                                                                           frame_window)
            return new_trajectory_list

        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        pnp_one_Rt_to_scene = np.array(
            self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id - 2))
        pnp_two_Rt_to_scene = np.array(
            self._db_ops.get_camera_rt_to_scene(self.session_id, self._start_image_id - 1))

        maya_data_ops = MayaDataInterface()
        rotate_one, camera_c_one = maya_data_ops.camera_rt_to_maya(pnp_one_Rt_to_scene)
        rotate_two, camera_c_two = maya_data_ops.camera_rt_to_maya(pnp_two_Rt_to_scene)
        assume_cam_distance = np.linalg.norm(camera_c_two - camera_c_one)

        twoview_ops = TwoViewCalculator()
        twoview_ops.set_focal_length(focal_length)
        twoview_ops.set_image_size(image_size)
        twoview_ops.initial_camera_intrinsic()
        twoview_ops.non_center_camera_matrix()
        sfm_images = self._db_ops.get_sfm_image_list(self.session_id)
        ba_start_idx = self._start_image_id - 2

        camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                  [0, focal_length, image_size[0] / 2.0],
                                  [0, 0, 1]])
        ba_update_from_image_id = self._start_image_id
        for dealing_image_idx in range(self._start_image_id, self._end_image_id):
            # Calculate new camera pose
            last_rt_to_scene = self._db_ops.get_camera_rt_to_scene(self.session_id, dealing_image_idx - 1)
            twoview_ops.set_correspondences(sfm_images[dealing_image_idx - 1].dump_matches_to_next(),
                                            sfm_images[dealing_image_idx].dump_matches_to_pre())
            normal_rel_Rt = twoview_ops.calc_relative_camera_pose()
            estimate_rel_Rt = np.hstack([normal_rel_Rt[:, :3], assume_cam_distance * normal_rel_Rt[:, [3]]])
            relative_trans_matrix = np.vstack([estimate_rel_Rt, np.array([[0, 0, 0, 1]])])
            last_trans_matrix = np.vstack([last_rt_to_scene, np.array([0, 0, 0, 1])])
            estimate_trans_matrix = np.matmul(relative_trans_matrix, last_trans_matrix)
            # Bundle adjustment from begin to current frame
            if bundle_adjust:
                rt_for_ba_list = np.empty((dealing_image_idx - ba_start_idx + 1, 6))
                rt_matrices = np.empty((dealing_image_idx - ba_start_idx + 1, 3, 4))

                rt_matrices[dealing_image_idx - ba_start_idx] = estimate_trans_matrix[:3, :]
                rt_for_ba_list[dealing_image_idx - ba_start_idx] = rt_matrix_for_ba(estimate_trans_matrix[:3, :])
                trajectory_list = refresh_trajectory_list(self.session_id,
                                                          self.work_dir,
                                                          ba_start_idx,
                                                          self._end_image_id,
                                                          dealing_image_idx - ba_start_idx + 1)
                point_count_each = len(trajectory_list)
                print(f">>> Bundle adjustment with {point_count_each} trajectories")
                image_points_list = np.empty((dealing_image_idx - ba_start_idx + 1, point_count_each, 2))
                for i in range(point_count_each):
                    for j in range(dealing_image_idx - ba_start_idx + 1):
                        image_points_list[j, i] = trajectory_list[i].corr_image_points[j].position_2d

                for ba_data_idx,  ba_image_idx in enumerate(range(ba_start_idx, dealing_image_idx)):
                    rt_to_scene = np.array(self._db_ops.get_camera_rt_to_scene(self.session_id, ba_image_idx))
                    rt_matrices[ba_data_idx] = rt_to_scene
                    rt_for_ba_list[ba_data_idx] = rt_matrix_for_ba(rt_to_scene)

                homo_world_points = np.zeros((point_count_each, 4, 1))
                # Triangulate
                project_matrices = np.matmul(camera_matrix, rt_matrices)
                image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
                for i in range(point_count_each):
                    homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)

                world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)

                ba_ops = BundleAdjustment()
                image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
                cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rt_for_ba_list,
                                                                                                world_pts, image_point,
                                                                                                camera_matrix,
                                                                                                weight_threshold)

                for ba_data_idx, update_image_id in enumerate(range(ba_update_from_image_id, dealing_image_idx + 1)):
                    offset = ba_update_from_image_id - ba_start_idx
                    rt_to_scene = rt_to_scene_from_vec(cam_rveces[ba_data_idx + offset], cam_tveces[ba_data_idx + offset])
                    self._db_ops.set_camera_rt_to_scene(self.session_id, update_image_id, rt_to_scene)
                    print(f">>> Rt to scene saved of image id {update_image_id}")
            else:
                self._db_ops.set_camera_rt_to_scene(self.session_id, dealing_image_idx, estimate_trans_matrix[:3, :])

    def calculate_op_sfm_forward_incremental_ba(self,
                                                new_operation: bool,
                                                user_tagged_pnp_op_ids: list,
                                                sfm_operation_id='',
                                                bundle_adjust=True,
                                                weight_threshold=0.292
                                                ):

        def refresh_trajectory_list(session_id: str, work_dir: str, start_im_id: int, end_im_id: int,
                                    frame_window: int) -> List[SFMTrajectory]:
            traj_forward_ops = TrajectoryForwardMem(session_id=session_id, work_dir=work_dir,
                                                    start_image_id=start_im_id, end_image_id=end_im_id)
            traj_forward_ops.calculate_trajectory_forward()
            return traj_forward_ops.get_trajectories(start_im_id, frame_window)

        TrajectoryForwardMem.initial_data(self.session_id, self._db_ops)

        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)

        pnp_user_tagged_operation_id = user_tagged_pnp_op_ids[0]

        pnp_image_range = self._db_ops.get_op_affected_range(self.session_id, pnp_user_tagged_operation_id)
        assert pnp_image_range[1] == self._start_image_id - 1, "Start image ID need adjacent with pnp image ID"

        pnp_rt_to_scene_list = self._db_ops.get_op_rt_to_scene_list(self.session_id,
                                                                    pnp_user_tagged_operation_id,
                                                                    pnp_image_range)

        pnp_one_rt_to_scene = np.array(pnp_rt_to_scene_list[str(self._start_image_id - 2)])
        pnp_two_rt_to_scene = np.array(pnp_rt_to_scene_list[str(self._start_image_id - 1)])

        maya_data_ops = MayaDataInterface()
        _, camera_c_one = maya_data_ops.camera_rt_to_maya(pnp_one_rt_to_scene)
        _, camera_c_two = maya_data_ops.camera_rt_to_maya(pnp_two_rt_to_scene)
        assume_cam_distance = np.linalg.norm(camera_c_two - camera_c_one)

        two_view_ops = TwoViewCalculator()
        two_view_ops.set_focal_length(focal_length)
        two_view_ops.set_image_size(image_size)
        two_view_ops.initial_camera_intrinsic()
        two_view_ops.non_center_camera_matrix()
        sfm_images = self._db_ops.get_sfm_image_list(self.session_id)
        ba_start_idx = self._start_image_id - 2

        camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                  [0, focal_length, image_size[0] / 2.0],
                                  [0, 0, 1]])
        ba_update_from_image_id = self._start_image_id

        if new_operation:
            affected_image_id_range = sorted(list(range(self._start_image_id, self._end_image_id)) + pnp_image_range)
            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id,
                                                              SFM_OPERATION.SFM_FORWARD,
                                                              affected_image_id_range
                                                              )
        self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, pnp_image_range[0],
                                               np.array(pnp_rt_to_scene_list[str(pnp_image_range[0])]))
        self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, pnp_image_range[1],
                                               np.array(pnp_rt_to_scene_list[str(pnp_image_range[1])]))

        for dealing_image_idx in range(self._start_image_id, self._end_image_id):
            # Calculate new camera pose
            last_rt_to_scene = self._db_ops.get_op_rt_to_scene(self.session_id, sfm_operation_id,
                                                               dealing_image_idx - 1)
            two_view_ops.set_correspondences(sfm_images[dealing_image_idx - 1].dump_matches_to_next(),
                                             sfm_images[dealing_image_idx].dump_matches_to_pre())
            normal_rel_rt = two_view_ops.calc_relative_camera_pose()
            estimate_rel_rt = np.hstack([normal_rel_rt[:, :3], assume_cam_distance * normal_rel_rt[:, [3]]])
            relative_trans_matrix = np.vstack([estimate_rel_rt, np.array([[0, 0, 0, 1]])])
            last_trans_matrix = np.vstack([last_rt_to_scene, np.array([0, 0, 0, 1])])
            estimate_trans_matrix = np.matmul(relative_trans_matrix, last_trans_matrix)
            # Bundle adjustment from begin to current frame
            if bundle_adjust:
                rt_for_ba_list = np.empty((dealing_image_idx - ba_start_idx + 1, 6))
                rt_matrices = np.empty((dealing_image_idx - ba_start_idx + 1, 3, 4))

                rt_matrices[dealing_image_idx - ba_start_idx] = estimate_trans_matrix[:3, :]
                rt_for_ba_list[dealing_image_idx - ba_start_idx] = rt_matrix_for_ba(estimate_trans_matrix[:3, :])
                trajectory_list = refresh_trajectory_list(self.session_id,
                                                          self.work_dir,
                                                          ba_start_idx,
                                                          self._end_image_id,
                                                          dealing_image_idx - ba_start_idx + 1)
                point_count_each = len(trajectory_list)
                print(f">>> Bundle adjustment with {point_count_each} trajectories")
                image_points_list = np.empty((dealing_image_idx - ba_start_idx + 1, point_count_each, 2))
                for i in range(point_count_each):
                    for j in range(dealing_image_idx - ba_start_idx + 1):
                        image_points_list[j, i] = trajectory_list[i].corr_image_points[j].position_2d

                for ba_data_idx,  ba_image_idx in enumerate(range(ba_start_idx, dealing_image_idx)):
                    rt_to_scene = np.array(
                        self._db_ops.get_op_rt_to_scene(self.session_id, sfm_operation_id, ba_image_idx))
                    rt_matrices[ba_data_idx] = rt_to_scene
                    rt_for_ba_list[ba_data_idx] = rt_matrix_for_ba(rt_to_scene)

                homo_world_points = np.zeros((point_count_each, 4, 1))
                # Triangulate
                project_matrices = np.matmul(camera_matrix, rt_matrices)
                image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
                for i in range(point_count_each):
                    homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)

                world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)

                ba_ops = BundleAdjustment()
                image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
                cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rt_for_ba_list,
                                                                                                world_pts, image_point,
                                                                                                camera_matrix,
                                                                                                weight_threshold)

                for ba_data_idx, update_image_id in enumerate(range(ba_update_from_image_id, dealing_image_idx + 1)):
                    offset = ba_update_from_image_id - ba_start_idx
                    rt_to_scene = rt_to_scene_from_vec(cam_rveces[ba_data_idx + offset],
                                                       cam_tveces[ba_data_idx + offset])
                    self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, update_image_id,
                                                           rt_to_scene)
                    print(f">>> Rt to scene saved of image id {update_image_id}")
            else:
                self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, dealing_image_idx,
                                                       estimate_trans_matrix[:3, :])


class NViewSFMBackward(CalNodeBase):
    def __init__(self, node_name='NViewSFMBackward', session_id='', work_dir='', start_image_id=-1, frame_window=1,
                 end_image_id=-1):
        super(NViewSFMBackward, self).__init__(node_name, session_id, work_dir)
        self._start_image_id = start_image_id
        self._frame_window = frame_window
        self._end_image_id = end_image_id
        self._db_ops = MongodbInterface(self.work_dir)
        self.dependent_nodes = ['TrajectoryForward']

    def calculate_sfm_backward_incremental_ba(self,
                                              new_operation=True,
                                              sfm_operation_id='',
                                              use_operation_ids=None,
                                              bundle_adjust=True,
                                              weight_threshold=0.292,
                                              min_trajectories=8):
        """
        In this function self.__start_image_id means the first frame before PnP frames
        self.__frame_window means PnP frames ahead
        :param new_operation:
        :param sfm_operation_id:
        :param use_operation_ids: Merge rt_to_scence of different operations
        :param bundle_adjust:
        :param weight_threshold:
        :param min_trajectories:
        :return:
        """
        def refresh_trajectory_list(session_id: str, work_dir: str, start_image_id: int, end_image_id: int,
                                    frame_window: int):
            traj_backward_ops = TrajectoryBackward(session_id=session_id, work_dir=work_dir,
                                                   start_image_id=start_image_id, end_image_id=end_image_id)

            traj_backward_ops.calculate_trajectory_backward()
            new_trajectory_list = self._db_ops.get_backward_trajectory_list(self.session_id,
                                                                            start_image_id,
                                                                            frame_window)
            return new_trajectory_list

        focal_length, image_size, _, _ = self._db_ops.get_images_info(self.session_id)
        pnp_user_tagged_operation_id = use_operation_ids[0]
        pnp_frames = self._db_ops.get_op_affected_range(self.session_id, pnp_user_tagged_operation_id)
        pnp_rt_to_scene_list = self._db_ops.get_op_rt_to_scene_list(self.session_id,
                                                                    pnp_user_tagged_operation_id,
                                                                    pnp_frames)

        assert pnp_frames[0] == self._start_image_id + 1, "The sfm backward start image is not align to PnP image id"
        pnp_one_rt_to_scene = np.array(pnp_rt_to_scene_list[str(pnp_frames[0])])
        pnp_two_rt_to_scene = np.array(pnp_rt_to_scene_list[str(pnp_frames[1])])

        maya_data_ops = MayaDataInterface()
        _, camera_c_one = maya_data_ops.camera_rt_to_maya(pnp_one_rt_to_scene)
        _, camera_c_two = maya_data_ops.camera_rt_to_maya(pnp_two_rt_to_scene)
        assume_cam_distance = np.linalg.norm(camera_c_two - camera_c_one)

        two_view_ops = TwoViewCalculator()
        two_view_ops.set_focal_length(focal_length)
        two_view_ops.set_image_size(image_size)
        two_view_ops.initial_camera_intrinsic()
        two_view_ops.non_center_camera_matrix()
        sfm_images = self._db_ops.get_sfm_image_list(self.session_id)
        ba_start_idx = self._start_image_id + 2

        camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                  [0, focal_length, image_size[0] / 2.0],
                                  [0, 0, 1]])
        ba_update_from_image_id = self._start_image_id

        if new_operation:
            affected_image_id_range = sorted(list(range(self._start_image_id, self._end_image_id, -1)) + pnp_frames)
            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id,
                                                              SFM_OPERATION.SFM_BACKWARD,
                                                              affected_image_id_range
                                                              )

        self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, pnp_frames[0],
                                               np.array(pnp_rt_to_scene_list[str(pnp_frames[0])]))
        self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, pnp_frames[1],
                                               np.array(pnp_rt_to_scene_list[str(pnp_frames[1])]))

        for dealing_image_idx in range(self._start_image_id, self._end_image_id, -1):
            # Calculate new camera pose
            last_rt_to_scene = np.array(self._db_ops.get_op_rt_to_scene(self.session_id,
                                                                        sfm_operation_id,
                                                                        dealing_image_idx + 1)
                                        )

            two_view_ops.set_correspondences(sfm_images[dealing_image_idx + 1].dump_matches_to_pre(),
                                             sfm_images[dealing_image_idx].dump_matches_to_next())
            normal_rel_Rt = two_view_ops.calc_relative_camera_pose()
            estimate_rel_Rt = np.hstack([normal_rel_Rt[:, :3], assume_cam_distance * normal_rel_Rt[:, [3]]])
            relative_trans_matrix = np.vstack([estimate_rel_Rt, np.array([[0, 0, 0, 1]])])
            last_trans_matrix = np.vstack([last_rt_to_scene, np.array([0, 0, 0, 1])])
            estimate_trans_matrix = np.matmul(relative_trans_matrix, last_trans_matrix)
            # Bundle adjustment from begin to current frame
            if bundle_adjust:
                rt_for_ba_list = np.empty((ba_start_idx - dealing_image_idx + 1, 6))
                rt_matrices = np.empty((ba_start_idx - dealing_image_idx + 1, 3, 4))

                rt_matrices[ba_start_idx - dealing_image_idx] = estimate_trans_matrix[:3, :]
                rt_for_ba_list[ba_start_idx - dealing_image_idx] = rt_matrix_for_ba(estimate_trans_matrix[:3, :])
                trajectory_list = refresh_trajectory_list(self.session_id,
                                                          self.work_dir,
                                                          ba_start_idx,
                                                          self._end_image_id,
                                                          ba_start_idx - dealing_image_idx + 1)
                point_count_each = len(trajectory_list)
                print(f">>> Bundle adjustment with {point_count_each} trajectories")
                image_points_list = np.empty((ba_start_idx - dealing_image_idx + 1, point_count_each, 2))
                for i in range(point_count_each):
                    for j in range(ba_start_idx - dealing_image_idx + 1):
                        image_points_list[j, i] = trajectory_list[i].corr_image_points[j].position_2d

                for ba_data_idx,  ba_image_idx in enumerate(range(ba_start_idx, dealing_image_idx, -1)):
                    rt_to_scene = np.array(self._db_ops.get_op_rt_to_scene(self.session_id,
                                                                           sfm_operation_id,
                                                                           ba_image_idx)
                                           )
                    rt_matrices[ba_data_idx] = rt_to_scene
                    rt_for_ba_list[ba_data_idx] = rt_matrix_for_ba(rt_to_scene)

                homo_world_points = np.zeros((point_count_each, 4, 1))
                # Triangulate
                project_matrices = np.matmul(camera_matrix, rt_matrices)
                image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
                for i in range(point_count_each):
                    homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)

                world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)

                ba_ops = BundleAdjustment()
                image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
                cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rt_for_ba_list,
                                                                                                world_pts, image_point,
                                                                                                camera_matrix,
                                                                                                weight_threshold)

                for ba_data_idx, update_image_id in enumerate(range(ba_update_from_image_id, dealing_image_idx - 1, -1)):
                    offset = ba_start_idx - ba_update_from_image_id
                    rt_to_scene = rt_to_scene_from_vec(cam_rveces[ba_data_idx + offset], cam_tveces[ba_data_idx + offset])
                    self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, update_image_id,
                                                           rt_to_scene)

                    print(f">>> Rt to scene saved of image id {update_image_id}")
            else:
                self._db_ops.set_operating_rt_to_scene(self.session_id, sfm_operation_id, dealing_image_idx,
                                                       estimate_trans_matrix[:3, :])


class TwoViewSFMBackward(CalNodeBase):
    def __init__(self, session_id='', node_name='TwoViewSFMBackward', work_dir='', start_image_id=-1):
        super(TwoViewSFMBackward, self).__init__(node_name, session_id, work_dir)
        self.__start_image_id = start_image_id


class TrajectoryForward(CalNodeBase):
    def __init__(self, node_name='TrajectoryForward', session_id='', work_dir='', start_image_id=-1, end_image_id=-1):
        super(TrajectoryForward, self).__init__(node_name, session_id, work_dir)
        assert -1 < start_image_id < end_image_id, "Image range define wrong!"
        self._start_image_id = start_image_id
        self._end_image_id = end_image_id
        self._db_ops = MongodbInterface(work_dir)
        self.dependent_nodes = ['CensusTrajectoryPairs']

    def calculate_trajectory_forward(self):
        """
        Use match_to_next in all sfm image data
        :return:
        """
        def last_section(current_trajectories: List[SFMTrajectory]) -> List[MatchedImagePoint]:
            if len(current_trajectories) > 0:
                last_section_points = list()
                for a_trajectory in current_trajectories:
                    last_section_points.append(a_trajectory.corr_image_points[-1])
                return last_section_points
            else:
                return []

        def forward_intersect_traj_indices(the_last_section_points: List[MatchedImagePoint]):
            the_last_image_id = 0
            for a_matched_point in the_last_section_points:
                if the_last_image_id < a_matched_point.sfm_image_id:
                    the_last_image_id = a_matched_point.sfm_image_id

            the_last_traj_idx_list = []
            for i, a_matched_point in enumerate(the_last_section_points):
                if a_matched_point.sfm_image_id == the_last_image_id:
                    the_last_traj_idx_list.append(i)

            return the_last_traj_idx_list

        trajectory_pairs_list = self._db_ops.get_trajectory_pairs(self.session_id)
        sfm_image_list = self._db_ops.get_sfm_image_list(self.session_id)
        self._db_ops.clear_forward_trajectories(self.session_id)

        trajectories = []

        curr_sfm_image = sfm_image_list[self._start_image_id]
        curr_trajectory_pairs = np.array(trajectory_pairs_list[self._start_image_id])
        next_trajectory_pairs = np.array(trajectory_pairs_list[self._start_image_id + 1])

        overlap_element = np.intersect1d(curr_trajectory_pairs[:, 1], next_trajectory_pairs[:, 0])

        if len(overlap_element) > 0:
            traj_start_candidates = overlap_element.astype(int)
            for a_traj_start in traj_start_candidates:
                new_trajectory = SFMTrajectory()
                new_trajectory.traject_length = 1
                new_trajectory.start_image_id = self._start_image_id
                new_trajectory.corr_image_points.append(curr_sfm_image.matches_to_next[a_traj_start])
                trajectories.append(new_trajectory)

        else:
            print(f"TrajectoryForward.calculate_trajectory_forward:\nNo trajectory start at {self._start_image_id}!")
            return

        for image_idx in range(self._start_image_id + 1, self._end_image_id):
            curr_sfm_image = sfm_image_list[image_idx]
            pre_trajectory_pairs = np.array(trajectory_pairs_list[image_idx - 1])
            curr_trajectory_pairs = np.array(trajectory_pairs_list[image_idx])

            last_section_list = last_section(trajectories)
            intersect_traj_indices = forward_intersect_traj_indices(last_section_list)

            overlap_element, indices_in_pre, indices_in_curr = np.intersect1d(pre_trajectory_pairs[:, 1],
                                                                              curr_trajectory_pairs[:, 0],
                                                                              return_indices=True)
            overlap_element = overlap_element.astype(int)
            for traj_idx in intersect_traj_indices:
                last_point_id = trajectories[traj_idx].corr_image_points[-1].point_id
                if last_point_id in overlap_element:
                    current_trajectory_pair_idx = indices_in_curr[np.where(overlap_element == last_point_id)]
                    idx_in_curr_match_to_next = int(curr_trajectory_pairs[current_trajectory_pair_idx, 0][0])
                    trajectories[traj_idx].corr_image_points.append(
                        curr_sfm_image.matches_to_pre[idx_in_curr_match_to_next])
                    trajectories[traj_idx].traject_length += 1

        self._db_ops.save_forward_trajectory_list(self.session_id, trajectories)

        return trajectories


class TrajectoryForwardMem(CalNodeBase):
    trajectory_pairs_list = []
    sfm_image_list: List[SFMImage] = []

    def __init__(self, node_name='TrajectoryForwardMem', session_id='', work_dir='', start_image_id=-1,
                 end_image_id=-1):
        super(TrajectoryForwardMem, self).__init__(node_name, session_id, work_dir)
        assert -1 < start_image_id < end_image_id, "Image range define wrong!"
        self._start_image_id = start_image_id
        self._end_image_id = end_image_id
        self._db_ops = MongodbInterface(work_dir)
        self.dependent_nodes = ['CensusTrajectoryPairs']
        self._trajectory_list: List[SFMTrajectory] = []

    @classmethod
    def initial_data(cls, session_id: str, db_ops):
        cls.trajectory_pairs_list = db_ops.get_trajectory_pairs(session_id)
        cls.sfm_image_list = db_ops.get_sfm_image_list(session_id)
        return cls

    def calculate_trajectory_forward(self):
        """
        Use match_to_next in all sfm image data
        :return:
        """
        def last_section(current_trajectories: List[SFMTrajectory]) -> List[MatchedImagePoint]:
            if len(current_trajectories) > 0:
                last_section_points = list()
                for a_trajectory in current_trajectories:
                    last_section_points.append(a_trajectory.corr_image_points[-1])
                return last_section_points
            else:
                return []

        def forward_intersect_traj_indices(the_last_section_points: List[MatchedImagePoint]):
            the_last_image_id = 0
            for a_matched_point in the_last_section_points:
                if the_last_image_id < a_matched_point.sfm_image_id:
                    the_last_image_id = a_matched_point.sfm_image_id

            the_last_traj_idx_list = []
            for i, a_matched_point in enumerate(the_last_section_points):
                if a_matched_point.sfm_image_id == the_last_image_id:
                    the_last_traj_idx_list.append(i)

            return the_last_traj_idx_list



        self._trajectory_list = []

        curr_sfm_image = self.sfm_image_list[self._start_image_id]
        curr_trajectory_pairs = np.array(self.trajectory_pairs_list[self._start_image_id])
        next_trajectory_pairs = np.array(self.trajectory_pairs_list[self._start_image_id + 1])

        overlap_element = np.intersect1d(curr_trajectory_pairs[:, 1], next_trajectory_pairs[:, 0])

        if len(overlap_element) > 0:
            traj_start_candidates = overlap_element.astype(int)
            for a_traj_start in traj_start_candidates:
                new_trajectory = SFMTrajectory()
                new_trajectory.traject_length = 1
                new_trajectory.start_image_id = self._start_image_id
                new_trajectory.corr_image_points.append(curr_sfm_image.matches_to_next[a_traj_start])
                self._trajectory_list.append(new_trajectory)

        else:
            print(f"TrajectoryForward.calculate_trajectory_forward:\nNo trajectory start at {self._start_image_id}!")
            return

        for image_idx in range(self._start_image_id + 1, self._end_image_id):
            curr_sfm_image = self.sfm_image_list[image_idx]
            pre_trajectory_pairs = np.array(self.trajectory_pairs_list[image_idx - 1])
            curr_trajectory_pairs = np.array(self.trajectory_pairs_list[image_idx])

            last_section_list = last_section(self._trajectory_list)
            intersect_traj_indices = forward_intersect_traj_indices(last_section_list)

            overlap_element, indices_in_pre, indices_in_curr = np.intersect1d(pre_trajectory_pairs[:, 1],
                                                                              curr_trajectory_pairs[:, 0],
                                                                              return_indices=True)
            overlap_element = overlap_element.astype(int)
            for traj_idx in intersect_traj_indices:
                last_point_id = self._trajectory_list[traj_idx].corr_image_points[-1].point_id
                if last_point_id in overlap_element:
                    current_trajectory_pair_idx = indices_in_curr[np.where(overlap_element == last_point_id)]
                    idx_in_curr_match_to_next = int(curr_trajectory_pairs[current_trajectory_pair_idx, 0][0])
                    self._trajectory_list[traj_idx].corr_image_points.append(
                        curr_sfm_image.matches_to_pre[idx_in_curr_match_to_next])
                    self._trajectory_list[traj_idx].traject_length += 1

        return self._trajectory_list

    def get_trajectories(self, start_im_id: int, min_traj_length: int):
        ret_traj_list = []
        for a_traj in self._trajectory_list:
            if a_traj.start_image_id == start_im_id and a_traj.traject_length >= min_traj_length:
                ret_traj_list.append(a_traj)

        return ret_traj_list


class TrajectoryBackward(CalNodeBase):
    def __init__(self, node_name='TrajectoryBackward', session_id='', work_dir='', start_image_id=-1, end_image_id=-1):
        super(TrajectoryBackward, self).__init__(node_name, session_id, work_dir)
        assert start_image_id > end_image_id
        self.__start_image_id = start_image_id
        self.__end_image_id = end_image_id
        self.__db_ops = MongodbInterface(work_dir)

    def calculate_trajectory_backward(self):
        """
        Use match_to_next in all sfm image data
        :return:
        """
        def last_section(current_trajectories: List[SFMTrajectory]) -> List[MatchedImagePoint]:
            if len(current_trajectories) > 0:
                last_section_points = list()
                for a_trajectory in current_trajectories:
                    last_section_points.append(a_trajectory.corr_image_points[-1])
                return last_section_points
            else:
                return []

        def backward_intersect_traj_indices(the_last_section_points: List[MatchedImagePoint]):
            the_last_image_id = self.__start_image_id
            for a_matched_point in the_last_section_points:
                if the_last_image_id > a_matched_point.sfm_image_id:
                    the_last_image_id = a_matched_point.sfm_image_id

            the_last_traj_idx_list = []
            for i, a_matched_point in enumerate(the_last_section_points):
                if a_matched_point.sfm_image_id == the_last_image_id:
                    the_last_traj_idx_list.append(i)

            return the_last_traj_idx_list

        trajectory_pairs_list = self.__db_ops.get_trajectory_pairs(self.session_id)
        sfm_image_list = self.__db_ops.get_sfm_image_list(self.session_id)
        self.__db_ops.clear_backward_trajectories(self.session_id)

        trajectories = []

        curr_sfm_image = sfm_image_list[self.__start_image_id]
        curr_trajectory_pairs = np.array(trajectory_pairs_list[self.__start_image_id])
        next_trajectory_pairs = np.array(trajectory_pairs_list[self.__start_image_id - 1])

        overlap_element = np.intersect1d(curr_trajectory_pairs[:, 0], next_trajectory_pairs[:, 1])

        if len(overlap_element) > 0:
            traj_start_condidates = overlap_element.astype(int)
            for a_traj_start in traj_start_condidates:
                new_trajectory = SFMTrajectory()
                new_trajectory.traject_length = 1
                new_trajectory.start_image_id = self.__start_image_id
                new_trajectory.corr_image_points.append(curr_sfm_image.matches_to_pre[a_traj_start])
                trajectories.append(new_trajectory)

        else:
            print(f"TrajectoryForward.calculate_trajectory_forward:\nNo trajectory start at {self.__start_image_id}!")
            return

        for image_idx in range(self.__start_image_id - 1, self.__end_image_id, -1):
            curr_sfm_image = sfm_image_list[image_idx]
            pre_trajectory_pairs = np.array(trajectory_pairs_list[image_idx + 1])
            curr_trajectory_pairs = np.array(trajectory_pairs_list[image_idx])

            last_section_list = last_section(trajectories)
            intersect_traj_indices = backward_intersect_traj_indices(last_section_list)

            overlap_element, indices_in_pre, indices_in_curr = np.intersect1d(pre_trajectory_pairs[:, 0],
                                                                              curr_trajectory_pairs[:, 1],
                                                                              return_indices=True)
            overlap_element = overlap_element.astype(int)
            for traj_idx in intersect_traj_indices:
                last_point_id = trajectories[traj_idx].corr_image_points[-1].point_id
                if last_point_id in overlap_element:
                    current_trajectory_pair_idx = indices_in_curr[np.where(overlap_element == last_point_id)]
                    idx_in_curr_match_to_next = int(curr_trajectory_pairs[current_trajectory_pair_idx, 0][0])
                    trajectories[traj_idx].corr_image_points.append(
                        curr_sfm_image.matches_to_pre[idx_in_curr_match_to_next])
                    trajectories[traj_idx].traject_length += 1

        self.__db_ops.save_backward_trajectory_list(self.session_id, trajectories)

        return trajectories


class GlobalBundleAdjustment(CalNodeBase):
    def __init__(self, session_id='', node_name='GlobalBundleAdjustment', work_dir='', start_image_id=-1, end_image_id=-1):
        super(GlobalBundleAdjustment, self).__init__(node_name, session_id, work_dir)
        self._start_image_id = start_image_id
        self._end_image_id = end_image_id
        self._ba_width = end_image_id - start_image_id
        self._db_ops = MongodbInterface(self.work_dir)
        self._trajectory_list = []

    def simple_global_bundle_adjustment(self, min_traj_count: int, update_from_image_id: int, weight_threshold=0.292,
                                        log_ba_result=False):

        def refresh_trajectory_list(session_id: str, work_dir: str, start_image_id: int, end_image_id: int,
                                    frame_window: int):
            traj_forward_ops = TrajectoryForward(session_id=session_id, work_dir=work_dir,
                                                 start_image_id=start_image_id, end_image_id=end_image_id)

            traj_forward_ops.calculate_trajectory_forward()
            new_trajectory_list = self._db_ops.get_forward_trajectory_list(self.session_id,
                                                                           start_image_id,
                                                                           frame_window)
            return new_trajectory_list

        self._trajectory_list = refresh_trajectory_list(self.session_id,
                                                        self.work_dir,
                                                        self._start_image_id,
                                                        self._end_image_id,
                                                        self._ba_width)

        point_count_each = len(self._trajectory_list)
        print(f"With trajectory {point_count_each}")
        rts_for_ba = np.zeros((self._ba_width, 6))
        rt_matrices = np.zeros((self._ba_width, 3, 4))
        image_points_list = np.zeros((self._ba_width, point_count_each, 2))

        for i, image_idx in enumerate(range(self._start_image_id, self._end_image_id)):
            rt_to_scene = self._db_ops.get_camera_rt_to_scene(self.session_id, image_idx)
            rt_matrices[i] = np.array(rt_to_scene)
            rts_for_ba[i] = rt_matrix_for_ba(np.array(rt_to_scene))

        for i in range(point_count_each):
            for j in range(self._ba_width):
                image_points_list[j, i] = self._trajectory_list[i].corr_image_points[j].position_2d

        # Triangulate
        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        homo_world_points = np.zeros((point_count_each, 4, 1))
        camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                  [0, focal_length, image_size[0] / 2.0],
                                  [0, 0, 1]])

        project_matrices = np.matmul(camera_matrix, rt_matrices)
        image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))

        for i in range(point_count_each):
            homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)
        world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)

        ba_ops = BundleAdjustment()
        image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
        cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rts_for_ba, world_pts,
                                                                                        image_point, camera_matrix,
                                                                                        weight_threshold)

        # Save ba result
        if log_ba_result:
            self._db_ops.clear_ba_residual(self.session_id)
            repro_camera_Rts = np.zeros((self._ba_width, 6))
            for idx in range(self._ba_width):
                repro_camera_Rts[idx] = np.hstack([cam_rveces[idx], cam_tveces[idx]])

            ba_mean, ba_stdvar = ba_ops.n_view_residual(repro_camera_Rts, ba_world_pts, image_point, camera_matrix)
            ba_data_dict = {
                str(0): {
                    "mean": ba_mean,
                    "stdvar": ba_stdvar
                }
            }
            self._db_ops.save_ba_residual(self.session_id, BA_PHASE.BA_INCREMENTAL, ba_data_dict)
            print(f"---> Save current round BA mean {ba_mean} , standard variation {ba_stdvar}")
            
            ba_mean_list, ba_stdvar_lsit = ba_ops.n_view_residual(repro_camera_Rts, ba_world_pts, image_point, camera_matrix, False)
            self._db_ops.clear_ba_residual_perimage(self.session_id)
            for i, image_idx in enumerate(range(self._start_image_id, self._end_image_id)):
                ba_data_dict = {
                    "0": {
                        "mean": ba_mean_list[i],
                        "stdvar": ba_stdvar_lsit[i]
                        }
                    }
                self._db_ops.save_ba_residual_perimage(self.session_id, image_idx, BA_PHASE.BA_INCREMENTAL, ba_data_dict)

            delete_count = self._db_ops.clear_last_world_points(self.session_id)
            self._db_ops.save_world_points(self.session_id, BA_PHASE.BA_INCREMENTAL, ba_world_pts)

        offset = update_from_image_id - self._start_image_id
        for result_idx, image_idx in enumerate(range(update_from_image_id, self._end_image_id)):
            rt_to_scene = rt_to_scene_from_vec(cam_rveces[result_idx + offset], cam_tveces[result_idx + offset])
            self._db_ops.set_camera_rt_to_scene(session_id=self.session_id, image_id=image_idx, rt_to_scene=rt_to_scene)
            print(f"Update Rt_to_scene of image {image_idx}, from result {result_idx + offset}")

    def simple_merge_ops_global_ba(self, new_operation: bool, sfm_operation_id: str, merge_operation_ids: list,
                                   update_image_ids: list, min_traj_count: int = 8, weight_threshold=0.292,
                                   log_ba_result=False):

        def refresh_trajectory_list(session_id: str, work_dir: str, start_image_id: int, end_image_id: int,
                                    frame_window: int):
            traj_forward_ops = TrajectoryForward(session_id=session_id, work_dir=work_dir,
                                                 start_image_id=start_image_id, end_image_id=end_image_id)

            traj_forward_ops.calculate_trajectory_forward()
            new_trajectory_list = self._db_ops.get_forward_trajectory_list(self.session_id,
                                                                           start_image_id,
                                                                           frame_window)
            return new_trajectory_list

        affected_list_dict = self._db_ops.get_op_affected_range_list(self.session_id, merge_operation_ids)
        image_id_list = []
        for op_id, affected_range in affected_list_dict.items():
            image_id_list.extend(affected_range)
        merged_image_ids = list(set(image_id_list))

        assert self._start_image_id >= merged_image_ids[0] and self._end_image_id <= merged_image_ids[-1] + 1,\
            "Operation image out of required image IDs"

        self._trajectory_list = refresh_trajectory_list(self.session_id,
                                                        self.work_dir,
                                                        self._start_image_id,
                                                        self._end_image_id,
                                                        self._ba_width)

        point_count_each = len(self._trajectory_list)
        print(f"With trajectory {point_count_each}")
        rts_for_ba = np.zeros((self._ba_width, 6))
        rt_matrices = np.zeros((self._ba_width, 3, 4))
        image_points_list = np.zeros((self._ba_width, point_count_each, 2))

        ba_used_operation_id = merge_operation_ids[-1]
        simple_ba_affect_image_id_range = list(range(self._start_image_id, self._end_image_id))
        rt_to_scene_for_ba =\
            self._db_ops.get_op_rt_to_scene_list(self.session_id,
                                                 ba_used_operation_id,
                                                 simple_ba_affect_image_id_range)

        for i, image_idx in enumerate(simple_ba_affect_image_id_range):
            rt_to_scene = rt_to_scene_for_ba[str(image_idx)]
            rt_matrices[i] = np.array(rt_to_scene)
            rts_for_ba[i] = rt_matrix_for_ba(np.array(rt_to_scene))

        for i in range(point_count_each):
            for j in range(self._ba_width):
                image_points_list[j, i] = self._trajectory_list[i].corr_image_points[j].position_2d

        # Triangulate
        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        homo_world_points = np.zeros((point_count_each, 4, 1))
        camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                  [0, focal_length, image_size[0] / 2.0],
                                  [0, 0, 1]])

        project_matrices = np.matmul(camera_matrix, rt_matrices)
        image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))

        for i in range(point_count_each):
            homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)
        world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)

        ba_ops = BundleAdjustment()
        image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)
        cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rts_for_ba, world_pts,
                                                                                        image_point, camera_matrix,
                                                                                        weight_threshold)
        if new_operation:
            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id,
                                                              SFM_OPERATION.SIMPLE_BUNDLE_ADJUSTMENT,
                                                              update_image_ids)

        # Save ba result
        if log_ba_result:
            self._db_ops.clear_op_ba_residual(self.session_id, sfm_operation_id)
            repro_camera_rts = np.zeros((self._ba_width, 6))
            for idx in range(self._ba_width):
                repro_camera_rts[idx] = np.hstack([cam_rveces[idx], cam_tveces[idx]])

            ba_mean, ba_stdvar = ba_ops.n_view_residual(repro_camera_rts, ba_world_pts, image_point, camera_matrix)
            ba_data_dict = {
                str(0): {
                    "mean": ba_mean,
                    "stdvar": ba_stdvar
                }
            }
            self._db_ops.save_op_ba_residual(self.session_id, sfm_operation_id,
                                             SFM_OPERATION.SIMPLE_BUNDLE_ADJUSTMENT, ba_data_dict)
            print(f"---> Save current round BA mean {ba_mean} , standard variation {ba_stdvar}")

            ba_mean_list, ba_stdvar_lsit = ba_ops.n_view_residual(repro_camera_rts, ba_world_pts, image_point,
                                                                  camera_matrix, False)
            self._db_ops.clear_op_ba_residual_perimage(self.session_id, sfm_operation_id)
            images_benchmark_dict = {}
            for i, image_idx in enumerate(simple_ba_affect_image_id_range):
                ba_data_dict = {
                    str(0): {
                        "mean": ba_mean_list[i],
                        "stdvar": ba_stdvar_lsit[i]
                    }
                }
                images_benchmark_dict.setdefault(str(image_idx), ba_data_dict)

            self._db_ops.save_op_ba_residual_perimage(self.session_id,
                                                      sfm_operation_id,
                                                      SFM_OPERATION.SIMPLE_BUNDLE_ADJUSTMENT,
                                                      images_benchmark_dict)

        delete_count = self._db_ops.clear_op_last_world_points(self.session_id, sfm_operation_id)
        self._db_ops.save_op_last_world_points(self.session_id,
                                               sfm_operation_id,
                                               SFM_OPERATION.SIMPLE_BUNDLE_ADJUSTMENT,
                                               ba_world_pts)

        shifted_indices = get_shifted_range(list(range(0, self._ba_width)), update_image_ids,
                                            simple_ba_affect_image_id_range)
        rt_to_scene_dataset = {}
        for result_idx, image_idx in zip(shifted_indices, update_image_ids):
            rt_to_scene = rt_to_scene_from_vec(cam_rveces[result_idx], cam_tveces[result_idx])
            rt_to_scene_dataset[str(image_idx)] = rt_to_scene.tolist()
            print(f"Update Rt_to_scene of image {image_idx}, from result {result_idx}")

        self._db_ops.set_operating_rt_to_scene_list(self.session_id, sfm_operation_id, rt_to_scene_dataset)

        return sfm_operation_id

    def global_bundle_adjustment_iteratively(self,
                                             min_traj_count: int,
                                             update_from_image_id: int,
                                             iterative_num=100,
                                             weight_threshold=0.292,
                                             log_ba_result=False):



        point_count_each = len(self._trajectory_list)
        print(f"With trajectory {point_count_each}")
        rts_for_ba = np.zeros((self._ba_width, 6))
        image_points_list = np.zeros((self._ba_width, point_count_each, 2))
        rt_matrices = np.empty((self._ba_width, 3, 4))

        for i in range(point_count_each):
            for j in range(self._ba_width):
                image_points_list[j, i] = self._trajectory_list[i].corr_image_points[j].position_2d
        image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
        image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)

        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                  [0, focal_length, image_size[0] / 2.0],
                                  [0, 0, 1]])
        ba_count = 0
        world_pts = np.array(self._db_ops.get_last_world_points(self.session_id, BA_PHASE.BA_INCREMENTAL))
        # Triangulate
        for i, image_idx in enumerate(range(self._start_image_id, self._end_image_id)):
            rt_to_scene = self._db_ops.get_camera_rt_to_scene(self.session_id, image_idx)
            rt_matrices[i] = rt_to_scene
            rts_for_ba[i] = rt_matrix_for_ba(np.array(rt_to_scene))

        homo_world_points = np.zeros((point_count_each, 4, 1))
        # Try refresh triangulate world points on every iteration

        ba_ops = BundleAdjustment()
        offset = update_from_image_id - self._start_image_id

        self._db_ops.clear_ba_phase_residual(self.session_id, BA_PHASE.BA_GLOBAL_ITERATIVE)
        if log_ba_result:
            ba_data_dict = {}
            ba_data_dict_perimage = []
        while ba_count < iterative_num:
            ba_count += 1

            cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rts_for_ba, world_pts,
                                                                                            image_point, camera_matrix,
                                                                                            weight_threshold)
            for i in range(self._ba_width - offset):
                rts_for_ba[i + offset] = np.hstack([cam_rveces[i + offset], cam_tveces[i + offset]])
                rt_matrices[i + offset] = rt_to_scene_from_vec(cam_rveces[i + offset], cam_tveces[i + offset])
            # Log bundle adjustment
            if log_ba_result:
                ba_mean, ba_stdvar = ba_ops.n_view_residual(rts_for_ba, ba_world_pts, image_point, camera_matrix)
                ba_data_dict.setdefault(str(ba_count), {"mean": ba_mean, "stdvar": ba_stdvar})
                print(f"---> Save {ba_count} BA mean {ba_mean} , standard variation {ba_stdvar}")
                ba_mean_list, ba_stdvar_list = ba_ops.n_view_residual(rts_for_ba,
                                                                      ba_world_pts,
                                                                      image_point,
                                                                      camera_matrix,
                                                                      False)
                ba_data_per_round = []
                for i, image_idx in enumerate(range(self._start_image_id, self._end_image_id)):
                    data = {"mean": ba_mean_list[i], "stdvar": ba_stdvar_list[i]}
                    ba_data_per_round.append(data)

                ba_data_dict_perimage.append(ba_data_per_round)

            project_matrices = np.matmul(camera_matrix, rt_matrices)
            for i in range(point_count_each):
                homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)
            world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)
            print(f">>> Bundle count {ba_count}")

        if log_ba_result:
            self._db_ops.save_ba_residual(self.session_id, BA_PHASE.BA_GLOBAL_ITERATIVE, ba_data_dict)

            for i, image_idx in enumerate(range(self._start_image_id, self._end_image_id)):
                per_image_ba_data = dict()
                for j in range(ba_count):
                    per_image_ba_data.setdefault(str(j + 1), ba_data_dict_perimage[j][i])
                self._db_ops.save_ba_residual_perimage(self.session_id,
                                                       int(image_idx),
                                                       BA_PHASE.BA_GLOBAL_ITERATIVE,
                                                       per_image_ba_data)

            self._db_ops.save_world_points(self.session_id, BA_PHASE.BA_GLOBAL_ITERATIVE, ba_world_pts)

        for result_idx, image_idx in enumerate(range(update_from_image_id, self._end_image_id)):
            # rt_to_scene = rt_to_scene_from_vec(rts_for_ba[result_idx + offset][:3], rts_for_ba[result_idx + offset][3:])
            # self.__db_ops.set_camera_Rt_to_scene(session_id=self.session_id,
            #                                      image_id=image_idx,
            #                                      Rt_to_scene=rt_to_scene)
            self._db_ops.set_camera_rt_to_scene(session_id=self.session_id, image_id=image_idx,
                                                rt_to_scene=rt_matrices[result_idx + offset])
            print(f"Update Rt_to_scene of image {image_idx}, from result {result_idx + offset}")

    def merge_ops_global_ba_iteratively(self, new_operation: bool, sfm_operation_id: str,
                                        ordered_merge_operation_ids: list, update_image_ids: list, min_traj_count=8,
                                        iterative_num=100, weight_threshold=0.292, log_ba_result=False):

        affected_list_dict = self._db_ops.get_op_affected_range_list(self.session_id, ordered_merge_operation_ids)
        image_id_list = []
        for _, affected_range in affected_list_dict.items():
            image_id_list.extend(affected_range)
        merged_image_ids = list(set(image_id_list))

        assert self._start_image_id >= merged_image_ids[0] and self._end_image_id <= merged_image_ids[-1] + 1,\
            "Operation image out of required image IDs"

        assert self._start_image_id <= update_image_ids[0] and self._end_image_id >= update_image_ids[-1],\
            "Update image IDs out of range"

        point_count_each = len(self._trajectory_list)
        print(f"With trajectory {point_count_each}")
        rts_for_ba = np.zeros((self._ba_width, 6))
        image_points_list = np.zeros((self._ba_width, point_count_each, 2))
        rt_matrices = np.empty((self._ba_width, 3, 4))

        for i in range(point_count_each):
            for j in range(self._ba_width):
                image_points_list[j, i] = self._trajectory_list[i].corr_image_points[j].position_2d

        image_point_traj = np.transpose(image_points_list, axes=(1, 0, 2))
        image_point = image_points_list.reshape(image_points_list.shape[0] * image_points_list.shape[1], 2)

        focal_length, image_size, image_count, _ = self._db_ops.get_images_info(self.session_id)
        camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                                  [0, focal_length, image_size[0] / 2.0],
                                  [0, 0, 1]])

        iterative_ba_affect_image_id_range = list(range(self._start_image_id, self._end_image_id))

        simple_ba_op_id = ordered_merge_operation_ids[0]
        world_pts = np.array(self._db_ops.get_specify_last_world_points(self.session_id,
                                                                        simple_ba_op_id,
                                                                        SFM_OPERATION.SIMPLE_BUNDLE_ADJUSTMENT))

        merged_image_id_dict = self._db_ops.get_op_affected_range_list(self.session_id, ordered_merge_operation_ids)

        image_id_selection_dict = get_ordered_image_ids_selection(iterative_ba_affect_image_id_range,
                                                                  ordered_merge_operation_ids,
                                                                  merged_image_id_dict)
        base_rt_list = {}
        for op_id in ordered_merge_operation_ids:
            operating_rt_list = self._db_ops.get_op_rt_to_scene_list(self.session_id,
                                                                     sfm_operation_id=op_id,
                                                                     image_id_range=image_id_selection_dict[op_id])
            base_rt_list |= operating_rt_list

        # Prepare for Triangulate
        for i, image_idx in enumerate(iterative_ba_affect_image_id_range):
            rt_to_scene = base_rt_list[str(image_idx)]
            rt_matrices[i] = np.array(rt_to_scene)
            rts_for_ba[i] = rt_matrix_for_ba(np.array(rt_to_scene))

        if new_operation:
            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id, SFM_OPERATION.GLOBAL_BA_ITERATIVELY,
                                                              update_image_ids)

        ba_data_dict = {}
        ba_data_dict_per_image = []
        if log_ba_result:
            self._db_ops.clear_op_ba_residual(self.session_id, sfm_operation_id)

        ba_ops = BundleAdjustment()
        ba_count = 0
        homo_world_points = np.zeros((point_count_each, 4, 1))

        update_shifted_range = get_shifted_range(list(range(self._ba_width)), update_image_ids,
                                                 iterative_ba_affect_image_id_range)
        while ba_count < iterative_num:
            ba_count += 1
            cam_rveces, cam_tveces, ba_world_pts = ba_ops.n_view_ba_sparsity_weighted_resid(rts_for_ba, world_pts,
                                                                                            image_point, camera_matrix,
                                                                                            weight_threshold)
            for i in update_shifted_range:
                rts_for_ba[i] = np.hstack([cam_rveces[i], cam_tveces[i]])
                rt_matrices[i] = rt_to_scene_from_vec(cam_rveces[i], cam_tveces[i])
            # Log bundle adjustment
            if log_ba_result:
                ba_mean, ba_stdvar = ba_ops.n_view_residual(rts_for_ba, ba_world_pts, image_point, camera_matrix)
                ba_data_dict.setdefault(str(ba_count), {"mean": ba_mean, "stdvar": ba_stdvar})
                print(f"---> Save {ba_count} BA mean {ba_mean} , standard variation {ba_stdvar}")
                ba_mean_list, ba_stdvar_list = ba_ops.n_view_residual(rts_for_ba,
                                                                      ba_world_pts,
                                                                      image_point,
                                                                      camera_matrix,
                                                                      False)
                ba_data_per_round = []
                for i, image_idx in enumerate(iterative_ba_affect_image_id_range):
                    data = {"mean": ba_mean_list[i], "stdvar": ba_stdvar_list[i]}
                    ba_data_per_round.append(data)

                ba_data_dict_per_image.append(ba_data_per_round)

            project_matrices = np.matmul(camera_matrix, rt_matrices)
            for i in range(point_count_each):
                homo_world_points[i] = X_from_xP_nonlinear(image_point_traj[i], project_matrices, image_size)
            world_pts = np.squeeze(homo_world_points[:, :3, :], axis=2)
            print(f">>> Bundle count {ba_count}")

        if log_ba_result:
            self._db_ops.save_op_ba_residual(self.session_id,
                                             sfm_operation_id,
                                             SFM_OPERATION.GLOBAL_BA_ITERATIVELY,
                                             ba_data_dict)
            per_image_ba_dataset = {}
            for i, image_idx in enumerate(iterative_ba_affect_image_id_range):
                per_image_ba_data = {}
                for j in range(ba_count):
                    per_image_ba_data.setdefault(str(j + 1), ba_data_dict_per_image[j][i])
                per_image_ba_dataset.setdefault(str(image_idx), per_image_ba_data)

            self._db_ops.save_op_ba_residual_perimage(self.session_id,
                                                      sfm_operation_id,
                                                      SFM_OPERATION.GLOBAL_BA_ITERATIVELY,
                                                      per_image_ba_dataset)
            self._db_ops.save_op_last_world_points(self.session_id,
                                                   sfm_operation_id,
                                                   SFM_OPERATION.GLOBAL_BA_ITERATIVELY,
                                                   ba_world_pts)

        save_rt_to_scene_dict = {}
        for result_idx, image_idx in zip(update_shifted_range, update_image_ids):
            save_rt_to_scene_dict[str(image_idx)] = rt_matrices[result_idx].tolist()
            print(f"Update Rt_to_scene of image {image_idx}, from result {result_idx}")

        self._db_ops.set_operating_rt_to_scene_list(self.session_id, sfm_operation_id, save_rt_to_scene_dict)

        return sfm_operation_id
