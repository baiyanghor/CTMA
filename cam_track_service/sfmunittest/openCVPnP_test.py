import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
from SFMOperations.SFMDatabaseInterface import SFMDatabaseInterface
from SFMOperations.TwoViewCalculator import TwoViewCalculator
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_Update"
focalLength = 933.3352
height = 540.0
width = 960.0

twoview_ops = TwoViewCalculator()
twoview_ops.set_focal_length(focalLength)
twoview_ops.initial_camera_intrinsic()

sfm_db_ops = SFMDatabaseInterface(work_dir)
sfm_db_ops.initial_sfm_database()
ground_true_one_list = sfm_db_ops.get_ground_true(1)
world_points = ground_true_one_list.dump_world_points()
image_one_points = ground_true_one_list.dump_image_points()

ground_true_two_list = sfm_db_ops.get_ground_true(2)
image_two_points = ground_true_two_list.dump_image_points()

image_one_file = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_Update/UnitTest/openCVAPITest/test-image/sequence_test.0002.jpg"
image_two_file = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_Update/UnitTest/openCVAPITest/test-image/sequence_test.0003.jpg"
twoview_ops.set_image_size((height, width))
twoview_ops.set_correspondences(image_one_points, image_two_points)
Rt = twoview_ops.calc_relative_camera_pose()
print("\nCamera Pose From two images")
# print(Rt)
# print('---------------------------------------------')
twoview_mat = Rt[:,:3]
print(np.around(twoview_mat, decimals=3))

k = np.eye(3)
k[0,0] = focalLength
k[1,1] = focalLength
k[0,2] = width / 2.0
k[1,2] = height / 2.0

world_points = world_points.astype(float)
image_one_points = image_one_points.astype(float)
ret_one, rvec_one, tvec_one = cv.solvePnP(world_points[:4], image_one_points[:4], k, None, flags=cv.SOLVEPNP_IPPE)
image_two_points = image_two_points.astype(float)
ret_two, rvec_two, tvec_two = cv.solvePnP(world_points[:4], image_two_points[:4], k, None, flags=cv.SOLVEPNP_IPPE)
#
# assert ret_one and ret_two
#
# r_mat_one = cv.Rodrigues(rvec_one)[0]
# r_mat_two = cv.Rodrigues(rvec_two)[0]
# relative_mat = np.matmul(np.linalg.inv(r_mat_one), r_mat_two)
# print("\nRelative rotation from PnP")
# print(np.around(relative_mat, decimals=3))
# print("\nDistance")
# numerical_distance = np.linalg.norm(tvec_two - tvec_one)
# print(numerical_distance)
# print("Second Distance")
# new_distance_one = np.matmul(np.linalg.inv(r_mat_one), tvec_one)
# new_distance_two = np.matmul(np.linalg.inv(r_mat_two), tvec_two)
# second_ = np.linalg.norm(new_distance_two - new_distance_one)
# print(second_)
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("Two rotation")
# print(new_distance_one)
# print("------------------------------------------------------")
# print(r_mat_one)
# print("------------------------------------------------------")
# print(np.matmul(r_mat_one.T, r_mat_one))
# print("------------------------------------------------------")
# print(new_distance_two)
# print("------------------------------------------------------")
# print(r_mat_two)
# print("------------------------------------------------------")
# print(np.matmul(r_mat_two.T, r_mat_two))
#
# print("Inverse Check")
# diff = r_mat_one.T - np.linalg.inv(r_mat_one)
# print(diff)
# diff = r_mat_two.T - np.linalg.inv(r_mat_two)
# print(diff)
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#
#
#
# a = np.array([-50.385, 33.878, 106.214])
# b = np.array([-48.354, 33.878, 106.213])
# print("\nTrue distance")
# true_distance = np.linalg.norm(b - a)
# print(true_distance)
# print("Scale")
# print(f"{numerical_distance / true_distance:3.3f}")
#
# print("tvec_one")
# print(tvec_one)
#
# print("tvec_two")
# print(tvec_two)
#
# print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++"
#       f"\nCamera transform from PnP"
#       f"\n++++++++++++++++++++++++++++++++++++++++++++++++++")
#
# print(f"Camera one")
# roller = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
# r_mat_180x = roller.as_matrix()
# r_mat_gl = np.matmul(r_mat_180x, r_mat_one)
# view_matrix = np.vstack([np.hstack([r_mat_gl, -tvec_one]), np.array([0, 0, 0, 1])])
# cvToGL = np.eye(4, dtype=float)
# cvToGL[1, 1] = -1.0
# cvToGL[2, 2] = -1.0
# view_matrix_gl = np.matmul(cvToGL, view_matrix)
# maya_matrix = view_matrix_gl.T
# print("Maya Matrix for camera one")
# print(maya_matrix)
# np.savetxt('maya_matrix.txt', maya_matrix, fmt="%1.7e")