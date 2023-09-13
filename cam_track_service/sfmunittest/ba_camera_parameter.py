import cv2 as cv
import numpy as np
from SFMOperations.SFMBundleAdjustment import BundleAdjustment
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.MayaDataInterface import MayaDataInterface
from SFMOperations.OpenCVOperators import OpenCVOperators
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py39_for_dist"

ba_ops = BundleAdjustment(None)
db_ops = MongodbInterface(work_dir)

focalLength = 933.3352
height = 540.0
width = 960.0

k = np.eye(3)
k[0, 0] = focalLength
k[1, 1] = focalLength
k[0, 2] = width / 2.0
k[1, 2] = height / 2.0

true_world_points = db_ops.get_user_world_points(session_id="6194d494a834ebdb9f929d94")
true_image_points = db_ops.get_user_images_points(session_id="6194d494a834ebdb9f929d94")

# print(f"True world points: \n {true_world_points}")
# print(f"True image points: \n {true_image_points}")

ret_one, rvec_one, tvec_one = cv.solvePnP(true_world_points[4:], true_image_points[0, 4:], k, None,
                                          flags=cv.SOLVEPNP_IPPE)
ret_two, rvec_two, tvec_two = cv.solvePnP(true_world_points[4:], true_image_points[1, 4:], k, None,
                                          flags=cv.SOLVEPNP_IPPE)

ret_one, rvec_one, tvec_one, inliers_one = cv.solvePnPRansac(true_world_points, true_image_points[0],
                                                             k, None, rvec_one, tvec_one,
                                                             useExtrinsicGuess=True, iterationsCount=1000)
ret_two, rvec_two, tvec_two, inliers_two = cv.solvePnPRansac(true_world_points, true_image_points[1],
                                                             k, None, rvec_two, tvec_two,
                                                             useExtrinsicGuess=True, iterationsCount=1000)


residual_weight_threshold = 0.292
camera_Rts = np.empty((2, 6), dtype=float)
camera_Rts[0] = np.hstack([rvec_one.flatten(), tvec_one.flatten()])
camera_Rts[1] = np.hstack([rvec_two.flatten(), tvec_two.flatten()])

raw_shape = true_image_points.shape

true_image_points_reshaped = true_image_points.reshape(raw_shape[0] * raw_shape[1], raw_shape[2])

ba_camera_Rts_weighted = ba_ops.ba_on_camera_paramter_weighted(camera_Rts, true_world_points,
                                                               true_image_points_reshaped, k,
                                                               residual_weight_threshold)

maya_data_interface = MayaDataInterface()

cv_ops = OpenCVOperators()
Rt_one = cv_ops.cv_rt_to_matrix(ba_camera_Rts_weighted[0, :3], ba_camera_Rts_weighted[0, 3:])
Rt_two = cv_ops.cv_rt_to_matrix(ba_camera_Rts_weighted[1, :3], ba_camera_Rts_weighted[1, 3:])

maya_cam_one = maya_data_interface.camera_rt_to_maya(Rt_one)
maya_cam_two = maya_data_interface.camera_rt_to_maya(Rt_two)

maya_cam_rotation_one = np.array([-10.237, -19.763, 0.000])
maya_cam_center_one = np.array([-50.385, 33.878, 106.214])
maya_cam_rotation_two = np.array([-10.293, -18.837, 0.000])
maya_cam_center_two = np.array([-48.354, 33.878, 106.213])

before_ba_Rt_one = cv_ops.cv_rt_to_matrix(rvec_one, tvec_one)
before_ba_Rt_two = cv_ops.cv_rt_to_matrix(rvec_two, tvec_two)
before_ba_maya_cam_one = maya_data_interface.camera_rt_to_maya(before_ba_Rt_one)
before_ba_maya_cam_two = maya_data_interface.camera_rt_to_maya(before_ba_Rt_two)

# Before ba distance
bfba_r_distance_one = before_ba_maya_cam_one[0] - maya_cam_rotation_one
bfba_t_distance_one = before_ba_maya_cam_one[1] - maya_cam_center_one
bfba_r_distance_two = before_ba_maya_cam_two[0] - maya_cam_rotation_two
bfba_t_distance_two = before_ba_maya_cam_two[1] - maya_cam_center_two

# After ba distance
r_distance_one = maya_cam_one[0] - maya_cam_rotation_one
t_distance_one = maya_cam_one[1] - maya_cam_center_one
r_distance_two = maya_cam_two[0] - maya_cam_rotation_two
t_distance_two = maya_cam_two[1] - maya_cam_center_two


# Check camera one
print("Camera one:\n--Before bundle adjustment:")
print(f"----Distance of rotation\n----{bfba_r_distance_one}"
      f"\n----mean: {np.mean(bfba_r_distance_one)}")
print(f"----Distance of translate\n----{bfba_t_distance_one}"
      f"\n----mean: {np.mean(bfba_t_distance_one)}")
print(f"--After bundle adjustment:")
print(f"----Distance of rotation\n----{r_distance_one}"
      f"\n----mean: {np.mean(r_distance_one)}")
print(f"----Distance of translate\n----{t_distance_one}"
      f"\n----mean: {np.mean(t_distance_one)}\n\n")

# Check camera two
print(f"Camera two:\n--Before bundle adjustment:"
      f"\n----Distance of rotation\n----{bfba_r_distance_two}"
      f"\n----mean: {np.mean(bfba_r_distance_two)}"
      f"\n----Distance of translate\n----{bfba_t_distance_two}"
      f"\n----mean: {np.mean(bfba_t_distance_two)}"
      f"\n--After bundle adjustment"
      f"\n----Distance of rotation\n----{r_distance_two}"
      f"\n----mean: {np.mean(r_distance_two)}"
      f"\n----Distance of translate\n----{t_distance_two}"
      f"\n----mean: {np.mean(t_distance_two)}")




