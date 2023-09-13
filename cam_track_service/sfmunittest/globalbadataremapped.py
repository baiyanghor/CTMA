from SFMOperations.SFMGlobalBACensusRemapped import GlobalBACensusRemapped


sfm_work_dir = f"E:/VHQ/camera-track-study/track-with-model/CameraTrackModel" \
               f"Alignment_py310_operationally/cam_track_service"

session_id = '61ee21c8d6a0b5298d93d145'

global_ba_op = GlobalBACensusRemapped(session_id=session_id, work_dir=sfm_work_dir, start_image_id=1, end_image_id=20)

census_triangulate_op_ids = ["6209c0f10aede32be5f90cdc"]
census_traj_op_id = "61f5f6ed53a164075ceb9fb2"

global_ba_op.calculate_remapped_ba(census_triangulation_op_ids=census_triangulate_op_ids,
                                   census_trajectory_op_id=census_traj_op_id,
                                   anchor_image_ids=['1', '2'],
                                   new_operation=True,
                                   iterate_num=500)
