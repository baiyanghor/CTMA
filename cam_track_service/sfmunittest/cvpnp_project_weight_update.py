import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from SFMOperations.SFMDatabaseInterface import SFMDatabaseInterface
from SFMOperations.TwoViewCalculator import TwoViewCalculator
from SFMOperations.SFMBundleAdjustment import BundleAdjustment



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



r_mat_one = cv.Rodrigues(rvec_one)[0]
camera_center_one = -np.matmul(r_mat_one.T, tvec_one)
r_mat_two = cv.Rodrigues(rvec_two)[0]
camera_center_two = -np.matmul(r_mat_two.T, tvec_two)

maya_cam_rotation_one = np.array([-10.237, -19.763, 0.000])
maya_cam_center_one = np.array([-50.385, 33.878, 106.214])
maya_cam_rotation_two = np.array([-10.293, -18.837, 0.000])
maya_cam_center_two = np.array([-48.354, 33.878, 106.213])

project_matrix_one = np.matmul(k, np.hstack([r_mat_one, tvec_one]))
project_matrix_two = np.matmul(k, np.hstack([r_mat_two, tvec_two]))
# homo_world_pts = cv.triangulatePoints(project_matrix_one, project_matrix_two, image_one_points.T, image_two_points.T)
# # print(homo_world_pts)
# world_pts = homo_world_pts[0:3,:] / homo_world_pts[[3],:]
#
# print("\nTriangulate from opencv sfm")
# print(world_pts.T)

twoviewOps = TwoViewCalculator()
homo_my_world_pts = twoviewOps.triangulate_two_view(project_matrix_one, project_matrix_two, image_one_points,
                                                    image_two_points, (height, width))
my_world_pts = homo_my_world_pts.squeeze(axis=2)[:, 0:3]

baOps = BundleAdjustment()
camera_Rts = np.empty((2, 6), dtype=float)
camera_Rts[0] = np.hstack([rvec_one.ravel(), tvec_one.ravel()])
camera_Rts[1] = np.hstack([rvec_two.ravel(), tvec_two.ravel()])
image_pts = np.vstack([image_one_points, image_two_points])

weight_sample_count = 40

unweighted_camera_rotation_one = np.empty((0))
unweighted_camera_rotation_two = np.empty((0))
unweighted_camera_translate_one = np.empty((0))
unweighted_camera_translate_two = np.empty((0))
weighted_camera_rotation_one = np.empty((0))
weighted_camera_rotation_two = np.empty((0))
weighted_camera_translate_one = np.empty((0))
weighted_camera_translate_two = np.empty((0))
unweighted_world_point = np.empty((0))
weighted_world_point = np.empty((0))

compare_x_axis = np.linspace(0.1, 0.6, weight_sample_count)

for update_weight in compare_x_axis:
    weighted_result = \
        baOps.two_view_ba_with_sparsity_matrix_wieght_update(camera_Rts, my_world_pts, image_pts, k, update_weight)

    # cam_rvecs, cam_tvecs, ret_world_pts = unweighted_result
    # unweighted_camera_rotation = np.empty((2, 3))
    # unweighted_camera_translate = np.empty((2, 3))
    # i = 0
    # for rvec, tvec in zip(cam_rvecs, cam_tvecs):
    #     r_mat = cv.Rodrigues(rvec)[0]
    #     rotator_x_180 = Rotation.from_euler('x', 180, degrees=True)
    #     r_mat_x_180 = rotator_x_180.as_matrix()
    #     opengl_r_mat = np.matmul(r_mat.T, r_mat_x_180)
    #     rotator_gl = Rotation.from_matrix(opengl_r_mat)
    #     unweighted_camera_rotation[i] = rotator_gl.as_euler('xyz', degrees=True)
    #     cam_center = -np.matmul(r_mat.T, tvec.reshape(3,1))
    #     unweighted_camera_translate[i] = cam_center.flatten()
    #     i += 1

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

    # unweighted_camera_rotation_one = np.append(unweighted_camera_rotation_one,
    #                                            np.linalg.norm(maya_cam_rotation_one - unweighted_camera_rotation[0]))
    # unweighted_camera_rotation_two = np.append(unweighted_camera_rotation_two,
    #                                            np.linalg.norm(maya_cam_rotation_two - unweighted_camera_rotation[1]))
    #
    # unweighted_camera_translate_one = np.append(unweighted_camera_translate_one,
    #                                             np.linalg.norm(maya_cam_center_one - unweighted_camera_translate[0]))
    # unweighted_camera_translate_two = np.append(unweighted_camera_translate_two,
    #                                             np.linalg.norm(maya_cam_center_two - unweighted_camera_translate[1]))
    #
    # ba_distance = np.diag(cdist(ret_world_pts, world_points, 'euclidean'))
    # unweighted_world_point = np.append(unweighted_world_point, np.mean(ba_distance))

    weighted_camera_rotation_one = np.append(weighted_camera_rotation_one,
                                             np.linalg.norm(maya_cam_rotation_one - weighted_camera_rotation[0]))
    weighted_camera_rotation_two = np.append(weighted_camera_rotation_two,
                                             np.linalg.norm(maya_cam_rotation_two - weighted_camera_rotation[1]))

    weighted_camera_translate_one = np.append(weighted_camera_translate_one,
                                              np.linalg.norm(maya_cam_center_one - weighted_camera_translate[0]))
    weighted_camera_translate_two = np.append(weighted_camera_translate_two,
                                              np.linalg.norm(maya_cam_center_two - weighted_camera_translate[1]))

    ba_distance_weighted = np.diag(cdist(ret_world_pts_weighted, world_points, 'euclidean'))

    weighted_world_point = np.append(weighted_world_point,
                                     np.mean(ba_distance_weighted))

    print(f"weighted_camera_rotation_one: {weighted_camera_rotation_one}")
    print(f"weighted_camera_rotation_two: {weighted_camera_rotation_two}")
    print(f"weighted_camera_translate_one: {weighted_camera_translate_one}")
    print(f"weighted_camera_translate_two: {weighted_camera_translate_two}")
    print(f"weighted_world_point: {weighted_world_point}")


fig, axes = plt.subplots(5, 1, sharex=True)

for i in range(len(compare_x_axis)):
    axes[0].annotate(round(compare_x_axis[i], 3), (compare_x_axis[i], weighted_camera_rotation_one[i]))
    axes[1].annotate(round(compare_x_axis[i], 3), (compare_x_axis[i], weighted_camera_rotation_two[i]))
    axes[2].annotate(round(compare_x_axis[i], 3), (compare_x_axis[i], weighted_camera_translate_one[i]))
    axes[3].annotate(round(compare_x_axis[i], 3), (compare_x_axis[i], weighted_camera_translate_two[i]))
    axes[4].annotate(round(compare_x_axis[i], 3), (compare_x_axis[i], weighted_world_point[i]))

axes[0].set_title("Weight of Camera Rotation One")
axes[0].grid(True)
axes[0].plot(compare_x_axis, weighted_camera_rotation_one, 'k')

axes[1].set_title("Weight of Camera Rotation Two")
axes[1].grid(True)
axes[1].plot(compare_x_axis, weighted_camera_rotation_two, 'k')

axes[2].set_title("Weight of Camera Translate One")
axes[2].grid(True)
axes[2].plot(compare_x_axis, weighted_camera_translate_one, 'k')

axes[3].set_title("Weight of Camera Translate Two")
axes[3].grid(True)
axes[3].plot(compare_x_axis, weighted_camera_translate_two, 'k')

axes[4].set_title("Weight of World Points")
axes[4].grid(True)
axes[4].plot(compare_x_axis, weighted_world_point, 'k')

plt.show()




