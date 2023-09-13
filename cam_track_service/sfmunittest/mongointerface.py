from pprint import pprint as pp

from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.SFMDataTypes import SFMTrajectory

work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_operationally/cam_track_service"
session_id = '61ee21c8d6a0b5298d93d145'
sfm_operation_id = '61ea2edd02fc585f9304cb79'
db_op = MongodbInterface(work_dir)

to_merge_op_ids = ["61f24712ec6a5916e26cf19d", "61f24ab6447717201b1cb552", "61efd53d7eb5cd27d1c7b7b8"]
# frames, image_points, world_points = db_op.get_user_tagged_features(session_id=session_id,
#                                                                     new_sfm_operation=False,
#                                                                     sfm_operation_id=sfm_operation_id)
#
# print(frames)
# print(image_points.shape)
# print(world_points.shape)

# pnp_rts = db_op.get_specify_camera_rt_to_scene_list(session_id=session_id,
#                                                     sfm_operation_id=sfm_operation_id,
#                                                     image_id_range=[21, 22])
#
# print(pnp_rts)

# f = db_op.get_op_affected_range(session_id, sfm_operation_id)
#
# print(f)

# range_list = db_op.get_op_affected_range_list(session_id=session_id, sfm_operation_ids=op_list)
# print(range_list)

# traj_list = db_op.get_census_trajectory_list(session_id, 1, 20, 6)
#
# print(len(traj_list))
#
# pp(traj_list)
session_id = '61ee21c8d6a0b5298d93d145'
census_trajectory_op_id = "61f5f6ed53a164075ceb9fb2"
census_triangulate_op_id = "6209c0f10aede32be5f90cdc"
triangulated_traj_list, initial_rts_to_scene = db_op.get_triangulated_trajectories(session_id=session_id,
                                                                                   census_trajectory_op_id=census_trajectory_op_id,
                                                                                   census_triangulation_op_id=census_triangulate_op_id)

pp(triangulated_traj_list)
pp(initial_rts_to_scene)

