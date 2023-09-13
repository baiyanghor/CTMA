from typing import List, NamedTuple
import numpy as np


class SFMSessionType(NamedTuple):
    NONE = 0
    MEMORY = 1
    DATABASE = 2
    DISKFILE = 3


class SFM_STATUS(NamedTuple):
    MATCHED_FEATURE_SAVED = "matched_feature_saved"
    INITIAL_SFM_PAIR = "initial_sfm_pair"
    USER_TAGGED_FEATURE_DONE = "user_tagged_feature_done"
    FIRST_PNP_READY = "first_pnp_ready"
    TRAJECTORY_PAIRS_READY = "trajectory_pairs_ready"
    INITIAL_SFM_ROUND_READY = "initial_sfm_round_ready"


class SFM_OPERATION(NamedTuple):
    USER_TAG_POINTS = "user_tag_points"
    PNP_USER_TAGGED_FRAMES = "pnp_user_tagged_frames"
    MODIFY_TAGGED_POINTS = "modify_tagged_points"
    SFM_FORWARD = "sfm_forward"
    SFM_BACKWARD = "sfm_backward"
    CENSUS_TRAJECTORIES = "census_trajectories"
    CENSUS_TRIANGULATION = "census_triangulation"
    SIMPLE_BUNDLE_ADJUSTMENT = "simple_bundle_adjustment"
    GLOBAL_BA_ITERATIVELY = "global_ba_iteratively"
    GLOBAL_BA_CENSUS_REMAPPED = "global_ba_census_remapped"
    TRACK_USER_TAGGED_FEATURES = "track_user_tagged_features"


class BA_PHASE(NamedTuple):
    BA_INCREMENTAL = 'incremental_ba'
    BA_GLOBAL_ITERATIVE = 'global_iterative_ba'


class GroundTrue(object):
    def __init__(self):
        self.point_id: int = -1
        self.world_point_3d: np.ndarray = None
        self.image_point_2d: np.ndarray = None


class GroundTrueList(object):
    def __init__(self):
        self.ground_true_list: List[GroundTrue]= []

    def dump_world_points(self):
        world_point_list = []
        for one_ground_true in self.ground_true_list:
            world_point_list.append(one_ground_true.world_point_3d)
        return np.array(world_point_list, dtype=float)

    def dump_image_points(self):
        image_point_list = []
        for one_image_point in self.ground_true_list:
            image_point_list.append(one_image_point.image_point_2d)
        return np.array(image_point_list, dtype=float)


class MatchedImagePoint(object):
    def __init__(self):
        self.point_id: int = -1
        # Which image the point lies in
        self.sfm_image_id: int = -1
        self.world_point_id: int = -1
        self.position_2d: np.ndarray = None

    def __str__(self):
        return (f'\nPoint ID: {self.point_id}'
                f'\nImage ID: {self.sfm_image_id}'
                f'\nWorld Point ID: {self.world_point_id}'
                f'\nposition_2d: {self.position_2d}')


class SFMImage(object):
    def __init__(self):
        self.image_id: int = -1
        self.pre_image_id: int = -1
        self.next_image_id: int = -1
        self.image_file_name: str = ''
        self.mask_file_name: str = ''
        # OpenCV Image Size Structure (height, width, color_depth)
        self.image_size: tuple = ()
        self.matches_to_pre: List[MatchedImagePoint] = []
        self.matches_to_next: List[MatchedImagePoint] = []
        self.trajectory_pair = None
        # Keep temporary features for shadow area filter out
        self.keep_indices_to_pre: np.ndarray = None
        self.keep_indices_to_next: np.ndarray = None
        self.matched_keep_indices_to_pre: np.ndarray = None
        self.matched_keep_indices_to_next: np.ndarray = None
        # Index pair [index_of_match_to_previous, index_of_match_to_next]
        self.trajectory_pair: np.ndarray = None
        self.sfm_selected_traj_pairs = []

    def __str__(self):
        return (f'\nID: {self.image_id}'
                f'\npre_image_id:{self.pre_image_id}'
                f'\nnext_image_id:{self.next_image_id}'
                f'\nimage_file_name: {self.image_file_name}'
                f'\nImage Size: {self.image_size}'
                f'\nMatches Previous: {len(self.matches_to_pre)}'
                f'\nMatched Next: {len(self.matches_to_next)}')

    def dump_matches(self, matches_to: List[MatchedImagePoint]) -> np.ndarray:
        matched_list = []
        for matched_image_point in matches_to:
            matched_list.append(matched_image_point.position_2d)
        return np.array(matched_list)

    def dump_matches_to_pre(self):
        return self.dump_matches(self.matches_to_pre)

    def dump_matches_to_next(self):
        return self.dump_matches(self.matches_to_next)

    def dump_trajectory_pairs(self):
        print(f'\n Image ID: {self.image_id}')
        for pair in self.trajectory_pair:
            minus_idx = np.where(pair == -1)
            # print(minus_idx)
            if minus_idx[0].shape[0] > 0:
                if minus_idx[0][0] == 0:
                    print(None, self.matches_to_next[pair[1]].position_2d)
                elif minus_idx[0][0] == 1:
                    print(self.matches_to_pre[pair[0]].position_2d, None)
            else:
                print(pair)
                print(self.matches_to_pre[pair[0]].position_2d, self.matches_to_next[pair[1]].position_2d)

    def dump_keep_indices_to_pre(self):
        return self.keep_indices_to_pre

    def dump_keep_indices_to_next(self):
        return self.keep_indices_to_next

    def dump_matched_indices_to_pre(self):
        return self.matched_keep_indices_to_pre

    def dump_matched_indices_to_next(self):
        return self.matched_keep_indices_to_next


class SFMCamera(object):
    def __init__(self):
        self.camera_id: int
        self.image_id: int
        self.intrinsic: np.ndarray
        self.Rt_to_pre: np.ndarray
        self.Rt_to_cam_0: np.ndarray
        self.R_to_pre: np.ndarray
        self.euc_to_pre: np.ndarray
        self.R_to_cam_0: np.ndarray
        self.euc_to_cam_0: np.ndarray

    def __str__(self):
        return (f'\nCamera ID:{self.camera_id}'
                f'\nRt_to_pre :'
                f'\n {self.Rt_to_pre}'
                # f'\nRt_to_cam_0:'
                # f'\n {self.Rt_to_cam_0}'
                )

    def copy_euclidean_to_pre(self):
        self.R_to_pre = self.Rt_to_pre[:, 0:3]
        # self.euc_to_pre = -np.matmul(self.R_to_pre.T, self.Rt_to_pre[:, [3]])
        self.euc_to_pre = self.Rt_to_pre[:, [3]]

    def copy_euclidean_to_cam_0(self):
        self.R_to_cam_0 = self.Rt_to_cam_0[:, 0:3]
        # self.euc_to_cam_0 = -np.matmul(self.R_to_cam_0.T, self.Rt_to_cam_0[:, [3]])
        self.euc_to_cam_0 = self.Rt_to_cam_0[:, [3]]


class SFMWorldPoint(object):
    def __init__(self):
        self.world_point_id: int = -1
        self.camera_ids: List[int] = []
        # Homogeneous format X, Y, Z, W
        self.position_3d: np.ndarray = None

    def __str__(self):
        return (f'World Point ID: {self.world_point_id}'
                f'\nCamera IDs: {self.camera_ids}'
                f'\nWorld Position: {self.position_3d}')

    def __repr__(self):
        return (f'World Point ID: {self.world_point_id}'
                f'  Camera IDs: {self.camera_ids}'
                f'  World Position: {self.position_3d}')


class TowViewMatchedPoints(object):
    def __init__(self):
        self.matched_id: int = -1
        self.image_id_a: int = -1
        self.image_id_b: int = -1
        self.matches_a: List[MatchedImagePoint] = []
        self.matches_b: List[MatchedImagePoint] = []

    def __str__(self):
        return (f'\nMatch ID: {self.matched_id}'
                f'\nImage ID A: {self.image_id_a}'
                f'\nImage ID B: {self.image_id_b}'
                f'\nMatches A: {len(self.matches_a)}'
                f'\nMatches B: {len(self.matches_b)}')

    def dump_matches(self, matches_to: List[MatchedImagePoint]):
        matched_list = []
        for matched_image_point in matches_to:
            matched_list.append(matched_image_point.position_2d)
        return np.asarray(matched_list)

    def dump_matches_a(self):
        return self.dump_matches(self.matches_a)

    def dump_matches_b(self):
        return self.dump_matches(self.matches_b)


class SFMTrajectory(object):
    def __init__(self):
        self.db_traj_id: str = ''
        self.start_image_id = 0
        self.corr_image_points: List[MatchedImagePoint] = []
        self.corr_traj_pair: List[list] = []
        self.traject_length: int = 0
        self.world_point: SFMWorldPoint = None

    # def __str__(self):
    #     return (f'\nWorld Point ID: {self.world_point.world_point_id}'
    #             f'\nCamera IDs: {self.world_point.camera_ids}'
    #             f'\nPosition: {self.world_point.position_3d}'
    #             f'\nTrajectory length: {len(self.corres_image_points)}'
    #             f'\nStart image id: {self.start_image_id}')

    def __str__(self):
        return (f"\nTrajectory db id: {self.db_traj_id}"
                f"\nStart image id: {self.start_image_id}"
                f"\nTrajectory length: {self.traject_length}"
                f"\nCorresponding point count: {len(self.corr_image_points)}"
                f"\nWorld point: {self.world_point}")

    def __repr__(self):
        return (f"Start image id: {self.start_image_id}"
                f"  Trajectory length: {self.traject_length}"
                f"\nCorresponding point count: {len(self.corr_image_points)}"
                f"\n{[p.sfm_image_id for p in self.corr_image_points]}"
                f"\nWorld point:\n{self.world_point}"
                )

    def get_next_image_id(self):
        return self.start_image_id + self.traject_length


class SFMResultData(object):
    def __init__(self, in_min_trajectory_window=3):
        self.min_trajectory_window: int = 0
        self.cameras: List[SFMCamera] = []
        self.trajectories: List[SFMTrajectory] = []
        self.min_trajectory_window = in_min_trajectory_window

    def dump_two_view_sfm_data(self, initial_pair: tuple) -> (np.ndarray, np.ndarray, np.ndarray):
        if self.trajectories:
            if len(self.trajectories) > 0:
                temp_world_points_one = []
                temp_world_points_two = []
                temp_image_one_points = []
                temp_image_two_points = []

                for a_trajectory in self.trajectories:
                    for a_match_point in a_trajectory.corr_image_points:
                        if a_match_point.sfm_image_id == initial_pair[0]:
                            temp_world_points_one.append(a_trajectory.world_point.position_3d)
                            temp_image_one_points.append(a_match_point.position_2d)
                        if a_match_point.sfm_image_id == initial_pair[1]:
                            temp_world_points_two.append(a_trajectory.world_point.position_3d)
                            temp_image_two_points.append(a_match_point.position_2d)

                temp_world_points_one = np.array(temp_world_points_one)
                temp_world_points_two = np.array(temp_world_points_two)

                concat_mat = np.array([[1000000, 1000, 1, 0]])
                temp_check_wp_one = np.matmul(concat_mat, np.around(temp_world_points_one, decimals=0)).flatten()
                temp_check_wp_two = np.matmul(concat_mat, np.around(temp_world_points_two, decimals=0)).flatten()
                temp_image_one_points = np.array(temp_image_one_points)
                temp_image_two_points = np.array(temp_image_two_points)

                _, in_one_ind, in_two_ind = np.intersect1d(temp_check_wp_one, temp_check_wp_two, return_indices=True)

                select = min(len(in_one_ind), len(in_two_ind))

                world_points = temp_world_points_one[in_one_ind[:select]]

                image_one_points = temp_image_one_points[in_one_ind[:select]]
                image_two_points = temp_image_two_points[in_two_ind[:select]]

                return world_points, image_one_points, image_two_points
            else:
                return None, None, None
        else:
            return None, None, None


class GlobalSFMData(object):
    def __init__(self):
        self.cached_image_count = 0
        self.cached_image_count: int
        self.sequence_image_size: tuple
        self.sfm_images: List[SFMImage] = []
        self.match_relation: List[TowViewMatchedPoints] = []
        self.sfm_data: SFMResultData = None

    def __str__(self):
        return (f'\nMatch relation: {len(self.match_relation)}'
                f'\nImage Count: {len(self.sfm_images)}')

