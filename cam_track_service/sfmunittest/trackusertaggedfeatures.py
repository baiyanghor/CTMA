from pprint import pprint as pp
import cv2 as cv
import numpy as np
from SFMOperations.SFMTrackTaggedFeatures import TrackTaggedFeatures
from SFMOperations.SFMMongodbInterface import MongodbInterface

sfm_work_dir = f"E:/VHQ/camera-track-study/track-with-model/" \
               f"CameraTrackModelAlignment_py310_of_tagged_feature_tracking/cam_track_service"

sfm_start_image_id = 5
frames_required = 10
sfm_end_image_id = sfm_start_image_id + frames_required

the_session_id = "62209de0e0baa22ad271a4a8"

user_tag_op_id = "62209e9163df5862b1a01699"

census_op_id = "6221c036f2c294602abe7dfa"

# save_to = f"E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_of_tagged_feature_tracking" /
#           f"/trackusertaggedpointsvalidate/usertaggedimagevalidate"
#
db_ops = MongodbInterface(sfm_work_dir)
#
# user_tagged_image_ids, user_tagged_image_points, user_tagged_world_points = db_ops.get_user_tagged_features(
#     the_session_id, sfm_operation_id=user_tag_op_id)
sfm_image_list = db_ops.get_sfm_image_list(the_session_id)
#
# im_file_0 = sfm_image_list[user_tagged_image_ids[0]].image_file_name
# im_file_1 = sfm_image_list[user_tagged_image_ids[1]].image_file_name
# frame_num_0 = im_file_0.split('.')[-2]
# frame_num_1 = im_file_1.split('.')[-2]
# mat_0 = cv.imread(im_file_0)
# mat_1 = cv.imread(im_file_1)
#
# for i in range(len(user_tagged_world_points)):
#     mat_0 = cv.circle(mat_0, np.rint(user_tagged_image_points[0][i]).astype(int), 5, (0, 0, 255), 2)
#     mat_1 = cv.circle(mat_1, np.rint(user_tagged_image_points[1][i]).astype(int), 5, (0, 0, 255), 2)
#
# cv.imwrite(f"{save_to}/user_taged.{frame_num_0}.jpg", mat_0)
# cv.imwrite(f"{save_to}/user_taged.{frame_num_1}.jpg", mat_1)

track_ops = TrackTaggedFeatures(session_id=the_session_id,
                                start_image_id=sfm_start_image_id,
                                end_image_id=sfm_end_image_id)

traj_list, of_errors = track_ops.tracking_tagged_features_forward(user_tagged_op_id=user_tag_op_id, new_operation=True)

pp(traj_list)
print("-" * 40)
pp(of_errors)

save_to = f"E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_of_tagged_feature_tracking/" \
          f"trackusertaggedpointsvalidate/initialerrortagged"

for image_id in range(sfm_start_image_id - 2, sfm_end_image_id):
    file_name = sfm_image_list[image_id].image_file_name
    frame_number = file_name.split('.')[-2]
    im_mat = cv.imread(file_name)
    for i, a_traj in enumerate(traj_list):
        pos = np.rint(a_traj.corr_image_points[image_id - sfm_start_image_id + 2].position_2d).astype(int)
        im_mat = cv.circle(im_mat, pos, 5, (0, 0, 255), 2)
        if image_id >= sfm_start_image_id:
            im_mat = cv.putText(im_mat,
                                f"{of_errors[image_id - sfm_start_image_id][i]: .4f}",
                                (pos[0], pos[1] - 10),
                                cv.FONT_HERSHEY_PLAIN,
                                1,
                                (0, 255, 0),
                                2)

    cv.imwrite(f"{save_to}/initialerrortagged.{frame_number}.jpg", im_mat)

# save_to = f"E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_of_tagged_feature_tracking" /
#           f"/trackusertaggedpointsvalidate/taggedimages"
#
# db_ops = MongodbInterface(work_dir=sfm_work_dir)
#
# sfm_image_list = db_ops.get_sfm_image_list(the_session_id)
# traj_list = db_ops.get_census_trajectory_list(the_session_id,
#                                               census_op_id,
#                                               sfm_start_image_id - 2,
#                                               sfm_end_image_id,
#                                               frames_required + 2)
# pp(traj_list)

# for im_id in range(sfm_start_image_id - 2, sfm_end_image_id):
#     full_file_name = sfm_image_list[im_id].image_file_name
#     frame_number = full_file_name.split('.')[-2]
#     im_mat = cv.imread(full_file_name)
#     for a_traj in traj_list:
#         im_mat = cv.circle(im_mat,
#                            np.rint(a_traj.corr_image_points[im_id - sfm_start_image_id - 2].position_2d).astype(int),
#                            5, (0, 0, 255), 2)
#     cv.imwrite(save_to + '/user_tagged.' + frame_number + '.jpg', im_mat)
