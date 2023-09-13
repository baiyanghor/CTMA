import numpy as np
import cv2 as cv
from SFMOperations import SFMPipeline as SFMpl
from SFMOperations.SFMDatabaseInterface import SFMDatabaseInterface

# Construct image list
image_file_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_Update/Unit" \
                 "Test/openCVAPITest/test-image"
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_Update"
focalLength = 933.3352
SFM_pipeline = SFMpl.SFMPipeLine()
SFM_pipeline.set_images_dir(image_file_dir)
SFM_pipeline.define_camera_intrinsic(focalLength)

# Construct matched feature list
SFM_pipeline.construct_sfm_image_list()
SFM_pipeline.non_center_camera_matrix()
# Filter out features in shadow

SFM_pipeline.filter_feature_in_shadow(False)
# for a_sfm_image in SFM_pipeline.sfm_pipeline_data.sfm_images:
#     matches_to_pre = a_sfm_image.dump_matches_to_pre()
#     matches_to_next = a_sfm_image.dump_matches_to_next()
#
#     if len(matches_to_pre) > 0:
#         print(matches_to_pre[a_sfm_image.dump_matched_indices_to_pre()])
#     else:
#         print("[]")
#
#     if len(matches_to_next) > 0:
#         print(matches_to_next[a_sfm_image.dump_matched_indices_to_next()])
#     else:
#         print("[]")

# Seize beast matched initial image pair from calculation not database
# SFM_pipeline.get_initial_sfm_pair()

# Get user selected features and 3D points
sfm_db_ops = SFMDatabaseInterface(work_dir)
sfm_db_ops.initial_sfm_database()
ground_true_one_list = sfm_db_ops.get_ground_true(1)
world_points = ground_true_one_list.dump_world_points()
image_one_points = ground_true_one_list.dump_image_points()

ground_true_two_list = sfm_db_ops.get_ground_true(2)
image_two_points = ground_true_two_list.dump_image_points()

initial_pair = sfm_db_ops.get_sfm_initial_pair(1)
print(f"Initial pair: {initial_pair}")

# print(world_points)
# print(image_one_points)
# print(image_two_points)

# Calculate camera pose from openCVPnP method
ret_one, rvec_one, tvec_one, inliers_one = cv.solvePnPRansac(world_points, image_one_points, SFM_pipeline.get_cv_intrinsic(), None)
ret_two, rvec_two, tvec_two, inliers_two = cv.solvePnPRansac(world_points, image_two_points, SFM_pipeline.get_cv_intrinsic(), None)

print(rvec_one)
print(rvec_two)
print(tvec_one)
print(tvec_two)

r_mat_one = cv.Rodrigues(rvec_one)[0]
r_mat_two = cv.Rodrigues(rvec_two)[0]
camera_center_one = -np.matmul(r_mat_one.T, tvec_one)
camera_center_two = -np.matmul(r_mat_two.T, tvec_two)
print(f"Camera center one: \n{camera_center_one}")
print(f"Camera center one: \n{camera_center_two}")

# Merge openCVPnP data and two view triangulation result



# Bundle adjustment on initial image pair with features that not in shadow

# Visualize residual before and after bundle adjustment

# Visualize residual with and without shadow filter

