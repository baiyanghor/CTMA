import numpy as np
import cv2 as cv
from SFMOperations.FundamentalMatrixCalculations import RANSAC_FM, openCV_FM
from SFMOperations.VGGFunctions import X_from_xP_linear, X_from_xP_nonlinear
from .SFMUtilities import normalized_p_0


def up_y(points: np.ndarray, image_height) -> np.ndarray:
    homo_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transform = np.array([[1, 0, 0], [0, -1, image_height]])
    return np.matmul(transform, homo_points.T).T


class TwoViewCalculator(object):
    def __init__(self):
        self.FM_algorithm = RANSAC_FM
        # self.FM_algorithm = openCV_FM
        self.focal_length = 1.0
        self.points_a = None
        self.points_b = None
        self.points_a_to_p = None
        self.points_b_to_p = None
        self.image_size = None
        self.__intrinsic_matrix = None
        self.__cv_camera_matrix = None

    def centralize_pix(self, points_a: np.ndarray, points_b: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Transform image point reference to principle point
        :param points_a: k X 2
        :param points_b: K X 2
        :return: -point + [w/2, h/2]
        """
        width = self.image_size[1]
        height = self.image_size[0]

        home_points_a = np.hstack([points_a, np.ones((points_a.shape[0], 1))])
        home_points_b = np.hstack([points_b, np.ones((points_b.shape[0], 1))])
        transform = np.array([[1, 0, -width / 2.0], [0, 1, -height / 2.0]])
        centralized_points_a = np.matmul(transform, home_points_a.T).T
        centralized_points_b = np.matmul(transform, home_points_b.T).T

        return centralized_points_a, centralized_points_b

    def set_correspondences(self, points_a: np.ndarray, points_b: np.ndarray, is_to_principle=False):
        """

        :param points_a: k X 2
        :param points_b: k X 2
        :param is_to_principle:
        :return:
        """
        assert points_a.shape == points_b.shape
        self.points_a = points_a
        self.points_b = points_b
        points_a_to_principle, points_b_to_principle = self.centralize_pix(points_a, points_b)
        self.points_a_to_p = points_a_to_principle
        self.points_b_to_p = points_b_to_principle

    def set_focal_length(self, in_focal_length):
        self.focal_length = in_focal_length

    def set_image_size(self, in_image_size: tuple):
        """
        :param in_image_size: order from OpenCV, height, width, color_depth
        :return:
        """
        self.image_size = in_image_size

    def initial_camera_intrinsic(self):
        intrinsic_matrix = np.eye(3)
        intrinsic_matrix[0][0] = self.focal_length
        intrinsic_matrix[1][1] = self.focal_length
        self.__intrinsic_matrix = intrinsic_matrix

    def non_center_camera_matrix(self):
        self.__cv_camera_matrix = self.__intrinsic_matrix.copy()
        self.__cv_camera_matrix[0,2] = self.image_size[1] / 2.0
        self.__cv_camera_matrix[1,2] = self.image_size[0] / 2.0
        return self.__cv_camera_matrix


    def set_intrinsic(self, in_intrinsic: np.ndarray):
        self.__intrinsic_matrix = in_intrinsic

    def get_FM(self):
        if self.points_a_to_p.shape == self.points_b_to_p.shape:
            FM_Data = self.FM_algorithm(self.points_a_to_p, self.points_b_to_p)
        # if self.points_a.shape == self.points_b.shape:
        #     FM_Data = self.FM_algorithm(self.points_a, self.points_b)
            FM = FM_Data[0]
            if FM.shape == (3, 3):
                scale = FM[2, 2]
                fundamental_matrix = FM / scale
                return fundamental_matrix
            else:
                print("Debug: Fundamental matrix calculation failure!")
                return None
        else:
            print("Debug: Points data does not initialed yet!")
            return None

    def get_EM_From_FM(self, F, K):
        if isinstance(F, np.ndarray) and isinstance(K, np.ndarray):
            return np.matmul(K.T, np.matmul(F, K))
        else:
            print("Debug: Input data type error!")
            return None

    def calc_relative_camera_pose(self):
        if self.image_size is None:
            print("Debug: Image Size is not initialed yet!")
            return None
        homo_fundamental_matrix = self.get_FM()
        E = self.get_EM_From_FM(homo_fundamental_matrix, self.__intrinsic_matrix)
        self.non_center_camera_matrix()
        # E = self.get_EM_From_FM(homo_fundamental_matrix, self.__cv_camera_matrix)
        U, s, Vt = np.linalg.svd(E, full_matrices=True)
        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])

        R_1 = np.matmul(np.matmul(U, W), Vt)
        R_2 = np.matmul(np.matmul(U, W.T), Vt)
        u__3 = U[:, [2]]

        if np.linalg.det(R_1) < 0:
            R_1 = -R_1

        if np.linalg.det(R_2) < 0:
            R_2 = -R_2

        candidate_P_1s = [np.hstack([R_1, u__3]), np.hstack([R_1, -u__3]), np.hstack([R_2, u__3]), np.hstack([R_2, -u__3])]

        P_0 = np.hstack([np.eye(3), np.zeros((3,1))])
        # Check for right candidate
        world_pts = np.zeros((self.points_a.shape[0], 4))
        project_matrices = np.zeros((2, 3, 4))
        matched_pts = np.zeros((2, 2))
        choices = np.zeros((1,4))

        # Affine map check
        for i, P_1 in enumerate(candidate_P_1s):
            for j, point_pairs in enumerate(zip(self.points_a, self.points_b)):
                project_matrices[0] = np.matmul(self.__cv_camera_matrix, P_0)
                project_matrices[1] = np.matmul(self.__cv_camera_matrix, P_1)
                # project_matrices[0] = np.matmul(self.__intrinsic_matrix, P_0)
                # project_matrices[1] = np.matmul(self.__intrinsic_matrix, P_1)
                # project_matrices[0] = P_0
                # project_matrices[1] = P_1
                matched_pts[0] = point_pairs[0]
                matched_pts[1] = point_pairs[1]
                world_pts[j] = X_from_xP_linear(matched_pts, project_matrices, self.image_size)

            euc_positions = world_pts[:, 0:3] / world_pts[:, [3, 3, 3]]
            C = -np.matmul(P_1[:, 0:3].T, P_1[:, [3]]).reshape((1, 3))
            distance_to_C_1s = euc_positions - np.repeat(C, euc_positions.shape[0], axis=0)
            rotate_z = P_1[2, 0:3].reshape((1, 3))
            depth = np.matmul(rotate_z, distance_to_C_1s.T)

            choices[0, i] = np.sum(np.bitwise_and(euc_positions[:, 2] > 0, depth[0, :] > 0))

        good_Rt_index = np.argmax(choices)

        return candidate_P_1s[good_Rt_index]

    def triangulate_two_view_baseline_scaled(self, baseline_scale: float) -> np.ndarray:
        P_0 = np.matmul(self.__cv_camera_matrix, normalized_p_0())
        Rt = self.calc_relative_camera_pose()
        scaled_Rt = np.hstack([Rt[:, 0:3], (baseline_scale * Rt[:,3]).reshape((3,1))])
        P_1 = np.matmul(self.__cv_camera_matrix, scaled_Rt)
        world_points = []
        project_matrices = np.zeros((2, 3, 4))
        project_matrices[0] = P_0
        project_matrices[1] = P_1
        for point_0, point_1 in zip(self.points_a, self.points_b):
            pts_position_2d = np.vstack([point_0, point_1])
            homo_world_positions = X_from_xP_nonlinear(pts_position_2d, project_matrices, self.image_size)
            homo_world_positions = homo_world_positions / homo_world_positions[3]
            world_points.append(homo_world_positions)

        return np.array(world_points)

    def triangulate_two_view(self, P_0: np.ndarray, P_1: np.ndarray, image_points_0: np.ndarray,
                             image_points_1: np.ndarray, image_size: tuple):
        world_points = []
        project_matrices = np.zeros((2, 3, 4))
        project_matrices[0] = P_0
        project_matrices[1] = P_1
        for point_0, point_1 in zip(image_points_0, image_points_1):
            pts_position_2d = np.vstack([point_0, point_1])
            homo_world_positions = X_from_xP_nonlinear(pts_position_2d, project_matrices, image_size)
            world_points.append(homo_world_positions)

        return np.array(world_points)


    def cv_get_relative_camera_pose(self):
        self.initial_camera_intrinsic()
        camera_matrix = self.non_center_camera_matrix()
        # essential, mask = cv.findEssentialMat(self.points_a,
        #                                       self.points_b,
        #                                       camera_matrix,
        #                                       threshold=0.05,
        #                                       prob=0.99,
        #                                       maxIters=1000
        #                                       )

        ret_value, E, R, t, mask = cv.recoverPose(points1=self.points_a,
                                                  points2=self.points_b,
                                                  cameraMatrix1=camera_matrix,
                                                  distCoeffs1=None,
                                                  cameraMatrix2=camera_matrix,
                                                  distCoeffs2=None,
                                                  method=cv.RANSAC,
                                                  prob=0.99,
                                                  threshold=1.0)

        return np.hstack([R, t])










