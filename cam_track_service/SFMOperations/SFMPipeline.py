import cv2
import numpy as np
from scipy.spatial import KDTree
from typing import List 
from .SFMDataTypes import SFMSessionType, GlobalSFMData, SFMResultData, TowViewMatchedPoints, MatchedImagePoint,\
                            SFMImage, SFMCamera, SFMTrajectory, SFMWorldPoint
from ImageOperations.OpenCVDataFromFileList import OpenCVData
from FileOperations.ImageFileOperators import ImageFileOperators
from SFMMongodbInterface import MongodbInterface

from .TwoViewCalculator import TwoViewCalculator
from .VGGFunctions import X_from_xP_nonlinear
from .N_ViewCalculator import NViewCalculator
from .SFMDatabaseInterface import SFMDatabaseInterface
from .SFMBundleAdjustment import BundleAdjustment
from .SFMUtilities import normalized_p_0


def match_indices(image_index, max_index):
    a = image_index - 1
    if a < 0:
        a = 0
    b = image_index
    if b == max_index:
        b = image_index - 1
    return a, b


class SFMPipeLine(object):

    def __init__(self, in_work_dir: str, session_type=SFMSessionType.MEMORY):
        self.sfm_pipeline_data = GlobalSFMData()
        self.sfm_pipeline_data.sfm_data = SFMResultData()
        self.__opencv_ops = OpenCVData(in_work_dir)
        self.__image_file_ops = ImageFileOperators(in_work_dir)
        self.__work_dir = in_work_dir
        self.__image_file_dir = ''
        self.__sequence_intrinsic = None
        self.__sequence_image_size = None
        self.__trajectory_tolerance_pix = 2
        self.__trajectory_forward = True
        self.__minimum_image_required = 3
        self.__work_dir = ''
        self.__sfm_db_ops = None
        self.__cv_camera_matrix = None
        self.__use_mask = False
        self.__session_type = session_type
        self.__session_id = ''

    def set_use_mask(self, in_use_mask=False):
        self.__use_mask = in_use_mask

    def set_images_dir(self, in_image_file_dir=''):
        self.__image_file_dir = in_image_file_dir

    def set_image_size(self, in_image_size):
        self.__sequence_image_size = in_image_size

    def set_session_id(self, in_session_id: str):
        self.__session_id = in_session_id

    def define_camera_intrinsic(self, focal_length: float):
        intrinsic = np.eye(3)
        intrinsic[0, 0] = focal_length
        intrinsic[1, 1] = focal_length
        self.__sequence_intrinsic = intrinsic

    def non_center_camera_matrix(self):
        self.__cv_camera_matrix = self.__sequence_intrinsic.copy()
        self.__cv_camera_matrix[0, 2] = self.__sequence_image_size[1] / 2.0
        self.__cv_camera_matrix[1, 2] = self.__sequence_image_size[0] / 2.0

    def get_cv_intrinsic(self):
        return self.__cv_camera_matrix

    def construct_sfm_image_list(self, use_mask=False):
        image_file_names = self.__image_file_ops.getImageRawFileList(self.__image_file_dir)
        image_mat_list = self.__opencv_ops.loadImageDataList(image_file_names)
        image_count = len(image_mat_list)
        self.sfm_pipeline_data.cached_image_count = image_count
        if image_count < self.__minimum_image_required:
            print(f"Debug construct_sfm_image_list: Require as least {self.__minimum_image_required} images!")
            return None

        self.__sequence_image_size = self.__opencv_ops.getImageSize()
        self.sfm_pipeline_data.sequence_image_size = self.__sequence_image_size
        self.__opencv_ops.initialFeatureExtractor('ORB_OF')

        # Construct Matched Points
        self.sfm_pipeline_data.match_relation = []
        for i in range(image_count-1):
            points_a, points_b = self.__opencv_ops.getCorrespondencesFromTwoView(image_mat_list[i], image_mat_list[i + 1])
            assert len(points_a) == len(points_b)
            a_match_relation = TowViewMatchedPoints()
            a_match_relation.matched_id = i
            a_match_relation.image_id_a = i
            a_match_relation.image_id_b = i+1
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

            self.sfm_pipeline_data.match_relation.append(a_match_relation)

        # Construct Image Information list
        self.sfm_pipeline_data.sfm_images = []
        for i in range(image_count):
            new_sfm_image = SFMImage()
            new_sfm_image.image_id = i
            new_sfm_image.pre_image_id = i - 1
            new_sfm_image.next_image_id = i + 1
            new_sfm_image.image_file_name = image_file_names[i]
            new_sfm_image.image_size = self.__sequence_image_size
            new_sfm_image.matches_to_pre = []
            new_sfm_image.matches_to_next = []

            matches_pre_idx, matches_next_idx = match_indices(i, image_count-1)

            new_sfm_image.matches_to_pre =\
                self.sfm_pipeline_data.match_relation[matches_pre_idx].matches_b
            new_sfm_image.matches_to_next =\
                self.sfm_pipeline_data.match_relation[matches_next_idx].matches_a

            self.sfm_pipeline_data.sfm_images.append(new_sfm_image)

        # Deal with loop boundaries
        self.sfm_pipeline_data.sfm_images[0].matches_to_pre = []
        self.sfm_pipeline_data.sfm_images[-1].matches_to_next = []

        return self.sfm_pipeline_data

    def calculate_camera_Rt_to_pre(self):
        self.sfm_pipeline_data.sfm_data.cameras = []

        new_sfm_camera = SFMCamera()
        new_sfm_camera.camera_id = 0
        new_sfm_camera.image_id = 0
        new_sfm_camera.intrinsic = self.__sequence_intrinsic
        new_sfm_camera.Rt_to_pre = np.hstack([np.eye(3), np.zeros((3,1))])
        new_sfm_camera.copy_euclidean_to_pre()
        self.sfm_pipeline_data.sfm_data.cameras.append(new_sfm_camera)

        two_view_calculator = TwoViewCalculator()
        two_view_calculator.set_intrinsic(self.__sequence_intrinsic)

        for i in range(1, self.sfm_pipeline_data.cached_image_count):
            new_sfm_camera = SFMCamera()
            new_sfm_camera.camera_id = i
            new_sfm_camera.image_id = i
            new_sfm_camera.intrinsic = self.__sequence_intrinsic
            two_view_calculator.set_image_size(self.__sequence_image_size)
            two_view_calculator.set_correspondences(self.sfm_pipeline_data.sfm_images[i-1].dump_matches_to_next(),
                                                    self.sfm_pipeline_data.sfm_images[i].dump_matches_to_pre())
            new_sfm_camera.Rt_to_pre = two_view_calculator.calc_relative_camera_pose()
            new_sfm_camera.copy_euclidean_to_pre()
            self.sfm_pipeline_data.sfm_data.cameras.append(new_sfm_camera)

    def calculate_Rt_to_cam0(self):
        self.sfm_pipeline_data.sfm_data.cameras[0].Rt_to_cam_0 = self.sfm_pipeline_data.sfm_data.cameras[0].Rt_to_pre
        self.sfm_pipeline_data.sfm_data.cameras[0].copy_euclidean_to_cam_0()

        for current_camera_idx in range(1, self.sfm_pipeline_data.cached_image_count):
            homo_Rt_to_pre = np.vstack([self.sfm_pipeline_data.sfm_data.cameras[current_camera_idx].Rt_to_pre,
                                        np.array([[0,0,0,1]])])
            homo_last_Rt_to_cam_0 = np.vstack([self.sfm_pipeline_data.sfm_data.cameras[current_camera_idx-1].Rt_to_cam_0,
                                               np.array([[0,0,0,1]])])

            self.sfm_pipeline_data.sfm_data.cameras[current_camera_idx].Rt_to_cam_0 = np.matmul(homo_Rt_to_pre,
                                                                                                homo_last_Rt_to_cam_0)[0:3,:]
            self.sfm_pipeline_data.sfm_data.cameras[current_camera_idx].copy_euclidean_to_cam_0()

    def construct_trajectory_forward(self):
        ...

    def construct_trajectory_backward(self):
        ...

    def construct_image_trajectory_pairs(self, tolerance_pix=0):
        for sfm_image_id in range(1, self.sfm_pipeline_data.cached_image_count - 1):
            current_sfm_image = self.sfm_pipeline_data.sfm_images[sfm_image_id]
            new_trajectory_pair = list()
            matched_next_list = current_sfm_image.dump_matches_to_next()
            matched_pre_list = current_sfm_image.dump_matches_to_pre()

            if len(matched_pre_list) > 0 and len(matched_next_list) > 0:
                searcher = KDTree(matched_pre_list)
                if tolerance_pix == 0:
                    search_radius = self.__trajectory_tolerance_pix
                else:
                    search_radius = tolerance_pix
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
                current_sfm_image.trajectory_pair = np.asarray(new_trajectory_pair)
            else:
                current_sfm_image.trajectory_pair = None

        # Deal with first and last frame
        if self.sfm_pipeline_data.sfm_images[1].trajectory_pair is None:
            self.sfm_pipeline_data.sfm_images[0].trajectory_pair = None
        else:
            minus_ones = -1 * np.ones((self.sfm_pipeline_data.sfm_images[1].trajectory_pair.shape[0], 1), dtype=np.int)
            self.sfm_pipeline_data.sfm_images[0].trajectory_pair = \
                np.hstack([minus_ones, self.sfm_pipeline_data.sfm_images[1].trajectory_pair[:, [0]]])

        if self.sfm_pipeline_data.sfm_images[-2].trajectory_pair is None:
            self.sfm_pipeline_data.sfm_images[-1].trajectory_pair = None
        else:
            minus_ones = -1 * np.ones((self.sfm_pipeline_data.sfm_images[-2].trajectory_pair.shape[0], 1), dtype=np.int)
            self.sfm_pipeline_data.sfm_images[-1].trajectory_pair = \
                np.hstack([self.sfm_pipeline_data.sfm_images[-2].trajectory_pair[:, [1]], minus_ones])

    def construct_trajectories(self):
        """
        Construct trajectory table
        :return: None or section of trajectories
        """
        def last_section(current_trajectories: List[SFMTrajectory]) -> object:
            if len(current_trajectories) > 0:
                last_section_points = list()
                for a_trajectory in current_trajectories:
                    last_section_points.append(a_trajectory.corr_image_points[-1])
                return last_section_points
            else:
                return None

        def last_image_id(the_last_section_points: List[MatchedImagePoint]):
            the_last_image_id = 0
            for a_matched_point in the_last_section_points:
                if the_last_image_id < a_matched_point.sfm_image_id:
                    the_last_image_id = a_matched_point.sfm_image_id
            return the_last_image_id

        trajectories = list()
        for curr_sfm_image_idx in range(1,self.sfm_pipeline_data.cached_image_count):
            curr_sfm_image = self.sfm_pipeline_data.sfm_images[curr_sfm_image_idx]
            pre_sfm_image = self.sfm_pipeline_data.sfm_images[curr_sfm_image_idx-1]

            connected, pre_indices, next_indices = np.intersect1d(pre_sfm_image.trajectory_pair[:,1],
                                                                  curr_sfm_image.trajectory_pair[:,0],
                                                                  return_indices=True)
            if connected.shape[0] > 0:
                connected = connected.astype(int)
                new_last_section_points = last_section(trajectories)
                if new_last_section_points is None or last_image_id(new_last_section_points) < pre_sfm_image.image_id:
                    # print(connected)
                    for a_connection in connected:
                        new_trajectory = SFMTrajectory()
                        new_trajectory.world_point = SFMWorldPoint()
                        new_trajectory.corr_image_points = list()
                        new_trajectory.corr_image_points.append(pre_sfm_image.matches_to_next[a_connection])
                        new_trajectory.corr_image_points.append(curr_sfm_image.matches_to_pre[a_connection])
                        trajectories.append(new_trajectory)
                else:
                    for traj_id, last_point in enumerate(new_last_section_points):
                        if last_point.sfm_image_id == pre_sfm_image.image_id:
                            last_point_indices = pre_sfm_image.trajectory_pair[pre_sfm_image.trajectory_pair[:, 0] == last_point.point_id]
                            if last_point_indices.shape[0] > 0:
                                corres_next_point_id = last_point_indices[0,1]
                                new_connect_point_id = connected[connected == corres_next_point_id]
                                if new_connect_point_id.shape[0] > 0:
                                    trajectories[traj_id].corr_image_points.append(curr_sfm_image.matches_to_pre[new_connect_point_id[0]])

        self.sfm_pipeline_data.sfm_data.trajectories = trajectories

    def filter_feature_in_shadow(self, refresh_mask=True):
        if refresh_mask:
            mask_image_files = self.__opencv_ops.generate_shadow_mask_list()
        else:
            mask_image_files = self.__opencv_ops.get_mask_image_filenames()

        for a_sfm_image, mask_file_name in zip(self.sfm_pipeline_data.sfm_images, mask_image_files):
            a_sfm_image.mask_file_name = mask_file_name
            mask_mat = cv2.imread(mask_file_name, flags=cv2.IMREAD_GRAYSCALE)
            features_to_pre = a_sfm_image.dump_matches_to_pre().astype("int")
            features_to_next = a_sfm_image.dump_matches_to_next().astype("int")

            if features_to_pre.shape[0] > 0:
                mask_values_to_pre = mask_mat[features_to_pre[:, 1], features_to_pre[:, 0]]
                keep_features_to_pre = np.where(mask_values_to_pre == 0)[0]
                a_sfm_image.keep_indices_to_pre = keep_features_to_pre
            else:
                a_sfm_image.keep_indices_to_pre = None

            if features_to_next.shape[0] > 0:
                mask_values_to_next = mask_mat[features_to_next[:, 1], features_to_next[:, 0]]
                keep_features_to_next = np.where(mask_values_to_next == 0)[0]
                a_sfm_image.keep_indices_to_next = keep_features_to_next
            else:
                a_sfm_image.keep_indices_to_next = None

        sfm_image_list = self.sfm_pipeline_data.sfm_images

        for i in range(len(sfm_image_list) - 1):
            keep_feature_0 = sfm_image_list[i].dump_keep_indices_to_next()
            keep_feature_1 = sfm_image_list[i+1].dump_keep_indices_to_pre()
            matched_keep_features = np.intersect1d(keep_feature_0, keep_feature_1)
            sfm_image_list[i].matched_keep_indices_to_next = matched_keep_features
            sfm_image_list[i+1].matched_keep_indices_to_pre = matched_keep_features


    def remove_fast_movements(self):
        ...

    def construct_triangulate_data(self):
        for guess_point_id, a_trajectory in enumerate(self.sfm_pipeline_data.sfm_data.trajectories):
            pts_position_2d = []
            project_matrices = []
            # Prepare data for nonlinear triangulating
            a_trajectory.world_point.camera_ids = []
            for a_matched_point in a_trajectory.corr_image_points:
                pts_position_2d.append(a_matched_point.position_2d)

                image_id = a_matched_point.sfm_image_id
                true_intrinsic = self.__sequence_intrinsic.copy()
                true_intrinsic[0, 2] = self.__sequence_image_size[1] / 2.0
                true_intrinsic[1, 2] = self.__sequence_image_size[0] / 2.0
                project_matrix = np.matmul(true_intrinsic, self.sfm_pipeline_data.sfm_data.cameras[image_id].Rt_to_cam_0)
                project_matrices.append(project_matrix)
                a_trajectory.world_point.camera_ids.append(image_id)
            # up_y_points_2d = up_y(np.asarray(pts_position_2d), self.sequence_image_size[0])
            homo_world_position = X_from_xP_nonlinear(np.asarray(pts_position_2d), np.asarray(project_matrices),
                                                      self.__sequence_image_size)

            a_trajectory.world_point.world_point_id = guess_point_id
            a_trajectory.world_point.position_3d = homo_world_position / homo_world_position[3]

    def incremental_bundle_adjustment(self):
        ...


    def initial_pair_bundle_adjustment(self):
        sfm_db_ops = SFMDatabaseInterface(self.__work_dir)
        sfm_db_ops.initial_sfm_database()
        initial_pair = sfm_db_ops.get_sfm_initial_pair(in_session_id=1)

        two_view_ops = TwoViewCalculator()
        two_view_ops.set_intrinsic(self.__sequence_intrinsic)
        two_view_ops.set_image_size(self.__sequence_image_size)
        two_view_ops.non_center_camera_matrix()
        initial_image_pts_zero = self.sfm_pipeline_data.sfm_images[initial_pair[0]].dump_matches_to_next()
        initial_image_pts_one = self.sfm_pipeline_data.sfm_images[initial_pair[1]].dump_matches_to_pre()
        two_view_ops.set_correspondences(initial_image_pts_zero, initial_image_pts_one)

        initial_distance = self.get_intial_PnP_distance()
        two_view_world_pts = two_view_ops.triangulate_two_view_baseline_scaled(initial_distance)
        two_view_world_pts = np.squeeze(two_view_world_pts, axis=2)

        ba_ops = BundleAdjustment(self.sfm_pipeline_data)
        self.non_center_camera_matrix()
        # w_pts, i_pts_one, i_pts_two = self.sfm_pipeline_data.sfm_data.dump_two_view_sfm_data(initial_pair)
        Rt = two_view_ops.calc_relative_camera_pose()
        # w_pts = np.squeeze(w_pts, axis=2)
        # unhomo_w_pts = w_pts[:, 0:3]
        P_0 = np.matmul(self.__cv_camera_matrix, normalized_p_0())
        scaled_Rt = np.hstack([Rt[:, 0:3], initial_distance * Rt[:, [3]]])
        P_1 = np.matmul(self.__cv_camera_matrix, scaled_Rt)
        unhomo_w_pts = two_view_world_pts[:, 0:3]
        ba_distance, ba_world_points = ba_ops.two_view_bundle_adjustment(1.0,
                                                                         unhomo_w_pts,
                                                                         initial_image_pts_zero,
                                                                         initial_image_pts_one,
                                                                         P_0,
                                                                         P_1)
        print("Camera distance after bundle adjustment")
        print(ba_distance)
        diff_world_pts = np.linalg.norm((ba_world_points - unhomo_w_pts), axis=1)
        max_distance_wpt_index = np.argmax(diff_world_pts)
        print("initial_pair_bundle_adjustment")
        print(np.sort(diff_world_pts)[::-1])
        wpts_variant = np.std(diff_world_pts)
        print("World points standard deviations")
        print(wpts_variant)
        return max_distance_wpt_index, unhomo_w_pts, ba_world_points

    def get_initial_sfm_pair(self):
        nvOps = NViewCalculator(self.sfm_pipeline_data)
        image_pair = nvOps.select_initial_triangulation_pair()

        if self.__session_type == SFMSessionType.DATABASE:
            with MongodbInterface(self.__work_dir) as db_ops:
                db_ops.save_sfm_initial_pair(self.__session_id, image_pair)

        return image_pair

    def get_intial_PnP_distance(self):
        sfm_db_ops = SFMDatabaseInterface(self.__work_dir)
        sfm_db_ops.initial_sfm_database()
        ground_true_one = sfm_db_ops.get_ground_true(1)
        ground_true_two = sfm_db_ops.get_ground_true(2)
        world_points = ground_true_one.dump_world_points()
        image_points_one = ground_true_one.dump_image_points()
        image_points_two = ground_true_two.dump_image_points()
        camera_intrinsic = self.__sequence_intrinsic.copy()
        camera_intrinsic[0, 2] = self.__sequence_image_size[1] / 2.0
        camera_intrinsic[1, 2] = self.__sequence_image_size[0] / 2.0
        ret_one, rvec_one, tvec_one = cv2.solvePnP(world_points[:4], image_points_one[:4],
                                                   camera_intrinsic, None, flags=cv2.SOLVEPNP_IPPE)
        ret_two, rvec_two, tvec_two = cv2.solvePnP(world_points[:4], image_points_two[:4],
                                                   camera_intrinsic, None, flags=cv2.SOLVEPNP_IPPE)
        r_mat_one = cv2.Rodrigues(rvec_one)[0]
        r_mat_two = cv2.Rodrigues(rvec_two)[0]
        c1 = np.matmul(r_mat_one.T, tvec_one)
        c2 = np.matmul(r_mat_two.T, tvec_two)

        distance = np.linalg.norm(c2 - c1)

        return distance

    def SFM_forward(self):
        ...


    def SFM_backward(self):
        ...


    def construct_feature_network(self):
        ...

    def write_sfm_result_to_database(self):
        ...