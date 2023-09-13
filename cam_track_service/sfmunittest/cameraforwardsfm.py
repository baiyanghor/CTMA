from SFMOperations.SFMCalculationNodes import NViewSFMForward, UserTaggedPNPExtrinsic

sfm_work_dir = f"E:/VHQ/camera-track-study/track-with-model/" \
               f"CameraTrackModelAlignment_py310_operationally/cam_track_service"

session_id = "61ee21c8d6a0b5298d93d145"
user_tag_op_id = "61f224de017c1bf21341c513"
pnp_user_op_id = "61f24712ec6a5916e26cf19d"

# pnp_ops = UserTaggedPNPExtrinsic(session_id=session_id, work_dir=sfm_work_dir)
# pnp_ops.calculate_tagged_pnp(new_sfm_operation=True, use_operation_ids=[user_tag_op_id])

sfm_forward_ops = NViewSFMForward(session_id=session_id, work_dir=sfm_work_dir, start_image_id=3, end_image_id=14)
sfm_forward_ops.calculate_op_sfm_forward_incremental_ba(new_operation=True, user_tagged_pnp_op_ids=[pnp_user_op_id],
                                                        bundle_adjust=True)



