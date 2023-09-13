from SFMOperations.SFMCalculationNodes import CensusTrajectoryPairs, TrajectoryBackward, UserTaggedPNPExtrinsic,\
    NViewSFMBackward, GlobalBundleAdjustment

work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_operationally/cam_track_service"
session_id = '61ee21c8d6a0b5298d93d145'
user_tag_operation_id = "61ee227aed0b4ea39b8547e8"
pnp_user_tagged_op_id = "61ee4e82b545b9c794a56922"
sfm_backward_op_id = "61ee561aa18145a825b2dfd5"
# censustraj_op = CensusTrajectoryPairs(session_id=session_id, work_dir=work_dir)
# censustraj_op.calculate_trajectories(0.9)

# traj_op = TrajectoryBackward(session_id=session_id, work_dir=work_dir, start_image_id=21, end_image_id=4)
# traj_op.calculate_trajectory_backward()

# user_tagged_pnp_op = UserTaggedPNPExtrinsic(session_id=session_id, work_dir=work_dir)
# user_tagged_pnp_op.calculate_tagged_pnp(new_sfm_operation=True, use_operation_ids=[user_tag_operation_id])

# sfm_backward_op = NViewSFMBackward(session_id=session_id, work_dir=work_dir, start_image_id=20, end_image_id=10)
# sfm_backward_op.calculate_sfm_backward_incremental_ba(new_operation=True, use_operation_ids=[pnp_user_tagged_op_id])

update_image_ids = list(range(11, 21))
ba_ops = GlobalBundleAdjustment(session_id=session_id, work_dir=work_dir, start_image_id=11, end_image_id=23)
simple_ba_op_id = ba_ops.simple_merge_ops_global_ba(new_operation=True, sfm_operation_id='', merge_operation_ids=[sfm_backward_op_id], update_image_ids=update_image_ids, log_ba_result=True)
ordered_operation_ids = [simple_ba_op_id, sfm_backward_op_id]
ba_ops.merge_ops_global_ba_iteratively(new_operation=True,
                                       sfm_operation_id='',
                                       ordered_merge_operation_ids=ordered_operation_ids,
                                       update_image_ids=update_image_ids,
                                       iterative_num=300,
                                       log_ba_result=True)








