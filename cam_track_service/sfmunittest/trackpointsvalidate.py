import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.SFMCalculationNodes import TrajectoryForward

work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/cam_track_service"
session_id = '61c12c6bb59e5e7b672ae3a0'

traj_fw_ops = TrajectoryForward(session_id=session_id, work_dir=work_dir, start_image_id=3, end_image_id=16)

traj_fw_ops.calculate_trajectory_forward()



# db_ops = MongodbInterface(work_dir)
#
# sfm_images_list = db_ops.get_sfm_image_list(session_id)
# trajectory_list = db_ops.get_trajectory_pairs(session_id)
# overlap, idx_in_pre_next, idx_in_next_pre = np.intersect1d(np.array(trajectory_list[2])[:,1], np.array(trajectory_list[3])[:,0], return_indices=True)
#
# print(overlap.shape)
# print(overlap)
#
# traj_next = np.array(trajectory_list[3])
# print(traj_next[idx_in_next_pre][:, 1])

# trajectory_ops = TrajectoryForward(session_id=session_id, work_dir=work_dir, start_image_id=2, end_image_id=7)
# traj_list = trajectory_ops.calculate_trajectory_forward()
#
# out_image_mat_zero = np.zeros((540, 960, 3), dtype='uint8')
# result = np.zeros((540, 960, 3), dtype='uint8')

# for i, a_traj in enumerate(traj_list):
#     print(f"{i} {a_traj.traject_length}")
#     out_image_mat = out_image_mat_zero.copy()
#     w = 1.0 / a_traj.traject_length
# a_traj = traj_list[5]
# for a_match_point in a_traj.corres_image_points:
#     image_mat = cv.imread(sfm_images_list[a_match_point.sfm_image_id].image_file_name)
#     color = np.random.randint(0, 255, 3).tolist()
#     cv.circle(image_mat, a_match_point.position_2d.astype('int').tolist(), 3, color, 2)
    # out_image_mat = cv.addWeighted(out_image_mat, w, image_mat, w, 0)

    # cv.imshow(f'Trajectory point {i} {a_traj.traject_length}', out_image_mat)
    # cv.imshow(f'Trajectory point {a_match_point.sfm_image_id}', image_mat.copy())

# result = cv.cvtColor(out_image_mat, cv.COLOR_BGR2RGB)
# plt.imshow(result)
# plt.show()
# cv.imshow(f'Trajectory point', out_image_mat)
# cv.waitKey()
#
#
# check_id_pair = (2, 3)
# image_mat_one = cv.imread(sfm_images_list[check_id_pair[0]].image_file_name)
# image_mat_two = cv.imread(sfm_images_list[check_id_pair[1]].image_file_name)
#
# trajectory_one = np.array(trajectory_list[check_id_pair[0]])
# trajectory_two = np.array(trajectory_list[check_id_pair[1]])
#
# image_one_to_pre = sfm_images_list[check_id_pair[0]].dump_matches_to_pre()
# image_one_to_next = sfm_images_list[check_id_pair[0]].dump_matches_to_next()
#
# image_two_to_pre = sfm_images_list[check_id_pair[1]].dump_matches_to_pre()
# image_two_to_next = sfm_images_list[check_id_pair[1]].dump_matches_to_next()
#
# intersect_p = np.intersect1d(trajectory_one[:, 1], trajectory_two[:, 0]).astype('int')

# for p in image_one_to_pre:
#     cv.circle(image_mat_one, p.astype('int'), 2, (200, 10, 10), 1)

# for p in image_one_to_next:
#     cv.circle(image_mat_one, p.astype('int'), 1, (10, 200, 200), 2)
# for p in image_one_to_pre:
#     cv.circle(image_mat_one, p.astype('int'), 1, (200, 0, 0), 2)

# for p in image_one_to_next[intersect_p]:
#     cv.circle(image_mat_one, p.astype('int'), 7, (200, 200, 200), 2)
#
# for p in image_two_to_pre:
#     cv.circle(image_mat_two, p.astype('int'), 3, (0, 0, 200), 3)

# for p in image_two_to_next:
#     cv.circle(image_mat_two, p.astype('int'), 1, (200, 0, 0), 2)

# for p in image_two_to_pre[intersect_p]:
#     cv.circle(image_mat_two, p.astype('int'), 7, (200, 200, 200), 2)
#
# cv.imshow('In image one', image_mat_one)
# cv.imshow('In image two', image_mat_two)
# cv.waitKey()
# result_image_one = cv.cvtColor(image_mat_one, cv.COLOR_BGR2RGB)
# result_image_two = cv.cvtColor(image_mat_two, cv.COLOR_BGR2RGB)
# fig, axes = plt.subplots(2, 1)
# axes[0].imshow(result_image_one)
# axes[1].imshow(result_image_two)
# plt.show()
