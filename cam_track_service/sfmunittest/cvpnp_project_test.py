import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from SFMOperations.SFMDatabaseInterface import SFMDatabaseInterface
from SFMOperations.TwoViewCalculator import TwoViewCalculator
from SFMOperations.SFMBundleAdjustment import BundleAdjustment

def project_matrix(k: np.ndarray, rvec: np.ndarray, tvec: np.ndarray):
    r_mat = cv.Rodrigues(rvec)[0]
    Rt = np.hstack([r_mat, tvec])
    return np.matmul(k, Rt)


def residual(pm: np.ndarray, world_pts: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    projected_image_pts = reproject(pm, world_pts)
    return np.linalg.norm((projected_image_pts - image_pts), axis=1)


def reproject(pm: np.ndarray, world_pts: np.ndarray):
    homo_world_pts = np.hstack([world_pts, np.ones(shape=(world_pts.shape[0], 1))])
    image_pts = np.matmul(pm, homo_world_pts.T)
    homo_image_pts = image_pts[[0, 1], :] / image_pts[[2], :]
    return homo_image_pts.T[:, [0, 1]]
    # return image_pts.T


work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_Update"
focalLength = 933.3352
height = 540.0
width = 960.0

k = np.eye(3)
k[0,0] = focalLength
k[1,1] = focalLength
k[0,2] = width / 2.0
k[1,2] = height / 2.0

sfm_db_ops = SFMDatabaseInterface(work_dir)
sfm_db_ops.initial_sfm_database()
ground_true_one_list = sfm_db_ops.get_ground_true(1)
world_points = ground_true_one_list.dump_world_points()
image_one_points = ground_true_one_list.dump_image_points()

ground_true_two_list = sfm_db_ops.get_ground_true(2)
image_two_points = ground_true_two_list.dump_image_points()

ret_one, rvec_one, tvec_one = cv.solvePnP(world_points[4:], image_one_points[4:], k, None, flags=cv.SOLVEPNP_IPPE)
ret_two, rvec_two, tvec_two = cv.solvePnP(world_points[4:], image_two_points[4:], k, None, flags=cv.SOLVEPNP_IPPE)

ret_one, rvec_one, tvec_one, inliers_one = cv.solvePnPRansac(world_points, image_one_points,
                                                             k, None, rvec_one, tvec_one,
                                                             useExtrinsicGuess=True, iterationsCount=1000)
ret_two, rvec_two, tvec_two, inliers_two = cv.solvePnPRansac(world_points, image_two_points,
                                                             k, None, rvec_two, tvec_two,
                                                             useExtrinsicGuess=True, iterationsCount=1000)

# print(world_points)
# print(image_one_points)
#
# print("================================================")
# print(rvec_one)
# print(tvec_one)
# print(inliers)
#
# pm = project_matrix(k, rvec_one, tvec_one)
# project_residual = residual(pm, world_points, image_one_points)
# print("------------------------------------------------")
# print(project_residual)

r_mat_one = cv.Rodrigues(rvec_one)[0]
camera_center_one = -np.matmul(r_mat_one.T, tvec_one)
r_mat_two = cv.Rodrigues(rvec_two)[0]
camera_center_two = -np.matmul(r_mat_two.T, tvec_two)
maya_cam_rotation_one = np.array([-10.237, -19.763, 0.000])
maya_cam_center_one = np.array([-50.385, 33.878, 106.214])
maya_cam_rotation_two = np.array([-10.293, -18.837, 0.000])
maya_cam_center_two = np.array([-48.354, 33.878, 106.213])


# offset_one = np.linalg.norm(camera_center_one.flatten() - maya_cam_center_one)
# offset_two = np.linalg.norm(camera_center_two.flatten() - maya_cam_center_two)
# print(offset_one)
# print(offset_two)
#
# mean_one = np.mean(np.vstack([camera_center_one, maya_cam_center_one]), axis=0)
# mean_two = np.mean(np.vstack([camera_center_two, maya_cam_center_two]), axis=0)
# print(mean_one)
# print(mean_two)

# print("\nRelative offset one")
# print((camera_center_one - maya_cam_center_one) / mean_one)
#
# print("\nRelative offset two")
# print((camera_center_two - maya_cam_center_two) / mean_two)

# Refine
# rvec_one, tvec_one = cv.solvePnPRefineLM(world_points, image_one_points, k, None, rvec_one, tvec_one)
# rvec_two, tvec_two = cv.solvePnPRefineLM(world_points, image_two_points, k, None, rvec_two, tvec_two)
# print("==========================================================")
# print("After refinement")
# r_mat_one = cv.Rodrigues(rvec_one)[0]
# camera_center_one = -np.matmul(r_mat_one.T, tvec_one)
# r_mat_two = cv.Rodrigues(rvec_two)[0]
# camera_center_two = -np.matmul(r_mat_two.T, tvec_two)
#
# camera_center_one = camera_center_one.ravel()
# camera_center_two = camera_center_two.ravel()
#
# offset_one = np.linalg.norm(camera_center_one.flatten() - maya_cam_center_one)
# offset_two = np.linalg.norm(camera_center_two.flatten() - maya_cam_center_two)
# print(offset_one)
# print(offset_two)
#
# mean_one = np.mean(np.vstack([camera_center_one, maya_cam_center_one]), axis=0)
# mean_two = np.mean(np.vstack([camera_center_two, maya_cam_center_two]), axis=0)
# print(mean_one)
# print(mean_two)
#
# print("\nRelative offset one")
# print((camera_center_one - maya_cam_center_one) / mean_one)
#
# print("\nRelative offset two")
# print((camera_center_two - maya_cam_center_two) / mean_two)

# print("=====================================================")
# print("\nCV2 Triangulateion test\n")
project_matrix_one = np.matmul(k, np.hstack([r_mat_one, tvec_one]))
project_matrix_two = np.matmul(k, np.hstack([r_mat_two, tvec_two]))
# homo_world_pts = cv.triangulatePoints(project_matrix_one, project_matrix_two, image_one_points.T, image_two_points.T)
# # print(homo_world_pts)
# world_pts = homo_world_pts[0:3,:] / homo_world_pts[[3],:]
#
# print("\nTriangulate from opencv sfm")
# print(world_pts.T)
print("\n============================================")
print("\nMy triangular")
twoviewOps = TwoViewCalculator()
homo_my_world_pts = twoviewOps.triangulate_two_view(project_matrix_one, project_matrix_two, image_one_points,
                                                    image_two_points, (height, width))
my_world_pts = homo_my_world_pts.squeeze(axis=2)[:, 0:3]
#
# print("\nImage points")
# print(image_one_points)
# print(image_two_points)
#
print("\n================================================")
print("Bundle adjustment test")
baOps = BundleAdjustment()
camera_Rts = np.empty((2, 6), dtype=float)
camera_Rts[0] = np.hstack([rvec_one.ravel(), tvec_one.ravel()])
camera_Rts[1] = np.hstack([rvec_two.ravel(), tvec_two.ravel()])
image_pts = np.vstack([image_one_points, image_two_points])
# print(camera_Rts)
# print(camera_Rts.shape)
# print("\n")
# print(my_world_pts)
# print(my_world_pts.shape)
# print('\n')
# print(image_pts)
# print(image_pts.shape)

# Centerlize image points
# image_pts[:, 0] = image_pts[:, 0] - width / 2.0
# image_pts[:, 1] = image_pts[:, 1] - height / 2.0
#
# print("\nCenter image points")
# print(image_pts)

# cam_rvecs, cam_tvecs, ba_world_pts = baOps.two_view_ba_with_sparsity_matrix(camera_Rts, my_world_pts, image_pts,
#                                                                             focalLength)

unweighted_result, weighted_result = baOps.two_view_ba_with_sparsity_matrix(camera_Rts, my_world_pts, image_pts,
                                                                            k)

print("---------------------------------------------------")
print("Unweighted BA results!\n")
cam_rvecs, cam_tvecs, ret_world_pts = unweighted_result
unweighted_camera_rotation = np.empty((2, 3))
unweighted_camera_translate = np.empty((2, 3))
i = 0
for rvec, tvec in zip(cam_rvecs, cam_tvecs):
    r_mat = cv.Rodrigues(rvec)[0]
    rotator_x_180 = Rotation.from_euler('x', 180, degrees=True)
    r_mat_x_180 = rotator_x_180.as_matrix()
    opengl_r_mat = np.matmul(r_mat.T, r_mat_x_180)
    rotator_gl = Rotation.from_matrix(opengl_r_mat)
    unweighted_camera_rotation[i] = rotator_gl.as_euler('xyz', degrees=True)
    cam_center = -np.matmul(r_mat.T, tvec.reshape(3,1))
    unweighted_camera_translate[i] = cam_center.flatten()
    i += 1

print("\nWorld Points")
print(ret_world_pts)

print("===================================================")
print("Weighted BA results!\n")
cam_rvecs_weighted, cam_tvecs_weighted, ret_world_pts_weighted = weighted_result
weighted_camera_rotation = np.empty((2, 3))
weighted_camera_translate = np.empty((2, 3))
i = 0
for rvec, tvec in zip(cam_rvecs_weighted, cam_tvecs_weighted):
    r_mat = cv.Rodrigues(rvec)[0]
    rotator_x_180 = Rotation.from_euler('x', 180, degrees=True)
    r_mat_x_180 = rotator_x_180.as_matrix()
    opengl_r_mat = np.matmul(r_mat.T, r_mat_x_180)
    rotator_gl = Rotation.from_matrix(opengl_r_mat)
    weighted_camera_rotation[i] = rotator_gl.as_euler('xyz', degrees=True)
    cam_center = -np.matmul(r_mat.T, tvec.reshape(3,1))
    weighted_camera_translate[i] = cam_center.flatten()
    i += 1

print("\nWorld Points")
print(ret_world_pts_weighted)





# baOps.two_view_ba_with_sparsity_matrix(camera_Rts, my_world_pts, image_pts,focalLength)

# ba_camera_Rts, ba_camera_Rts_weighted = baOps.ba_on_camera_paramter(camera_Rts, world_points, image_pts, k)

# print("----------------------------------------------------------------")
# print("Unweighted")
# for a_cam_parameter in ba_camera_Rts:
#     r_mat = cv.Rodrigues(a_cam_parameter[:3])[0]
#
#     rotator_x_180 = Rotation.from_euler('x', 180, degrees=True)
#     r_mat_x_180 = rotator_x_180.as_matrix()
#     opengl_r_mat = np.matmul(r_mat.T, r_mat_x_180)
#     rotator_gl = Rotation.from_matrix(opengl_r_mat)
#     print(rotator_gl.as_euler('xyz', degrees=True))
#     tvec = a_cam_parameter[3:]
#     cam_center = -np.matmul(r_mat.T, tvec.reshape(3, 1))
#     print(cam_center)
#
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("Weighted")
# for a_cam_parameter in ba_camera_Rts_weighted:
#     r_mat = cv.Rodrigues(a_cam_parameter[:3])[0]
#
#     rotator_x_180 = Rotation.from_euler('x', 180, degrees=True)
#     r_mat_x_180 = rotator_x_180.as_matrix()
#     opengl_r_mat = np.matmul(r_mat.T, r_mat_x_180)
#     rotator_gl = Rotation.from_matrix(opengl_r_mat)
#     print(rotator_gl.as_euler('xyz', degrees=True))
#     tvec = a_cam_parameter[3:]
#     cam_center = -np.matmul(r_mat.T, tvec.reshape(3, 1))
#     print(cam_center)

# print("\nBefore bundle adjustment Camera Center")
# print(camera_center_one)
# print(camera_center_two)
print("\n============================================")
print("\nGround True")
print("\nMaya Camera")
print(maya_cam_rotation_one)
print(maya_cam_center_one)
print(maya_cam_rotation_two)
print(maya_cam_center_two)
print("\nWorld Points")
print(world_points)

print("=============================================================")
print("Distance benchmarks")
print("-------------------------------------------------------------")
print("Unweighted")
print("\nCamera Rotation")
print(np.linalg.norm(maya_cam_rotation_one - unweighted_camera_rotation[0]))
print(np.linalg.norm(maya_cam_rotation_two - unweighted_camera_rotation[1]))
print("\nCamera Translate")
print(np.linalg.norm(maya_cam_center_one - unweighted_camera_translate[0]))
print(np.linalg.norm(maya_cam_center_two - unweighted_camera_translate[1]))
print("\nWorld Points")
ba_distance = np.diag(cdist(ret_world_pts, world_points, 'euclidean'))
# print(ba_distance)
print(np.mean(ba_distance))
print("--------------------------------------------------------------")
print("Weighted")
print("\nCamera Rotation")
print(np.linalg.norm(maya_cam_rotation_one - weighted_camera_rotation[0]))
print(np.linalg.norm(maya_cam_rotation_two - weighted_camera_rotation[1]))
print("\nCamera Translate")
print(np.linalg.norm(maya_cam_center_one - weighted_camera_translate[0]))
print(np.linalg.norm(maya_cam_center_two - weighted_camera_translate[1]))
print("\nWorld Points")
ba_distance_weighted = np.diag(cdist(ret_world_pts_weighted, world_points, 'euclidean'))
# print(ba_distance_weighted)
print(np.mean(ba_distance_weighted))



