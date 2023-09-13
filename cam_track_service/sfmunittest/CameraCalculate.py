import os
import cv2 as cv
from SFMOperations.SFMCalculationNodes import CensusTrajectoryPairs, NViewSFMForward, GlobalBundleAdjustment,\
    UserTaggedPNPExtrinsic

from SFMOperations.OpenCVOperators import OpenCVOperators
from SFMOperations.SFMMongodbInterface import MongodbInterface

sfm_work_dir = f"E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_operationally" \
               f"/cam_track_service"

session_id = '620b561a22e6af9a60695b81'

sfm_start_image_id = 3
sfm_end_image_id = 24  # Start image id and end image id have same mean of numpy array

user_tagged_op_id = "620b56ca6cfcfa55e5ba9085"

# census_traj_pair_op = CensusTrajectoryPairs(session_id=session_id, work_dir=sfm_work_dir)
#
# census_traj_pair_op.calculate_trajectories(pix_tolerance=0.9)

# user_tagged_pnp_op = UserTaggedPNPExtrinsic(session_id=session_id, work_dir=sfm_work_dir)
#
# user_tagged_pnp_op.calculate_tagged_pnp(use_operation_ids=[user_tagged_op_id], new_sfm_operation=True)

incremental_ba_frame = 5
n_view_sfm_forward_op = NViewSFMForward(session_id=session_id,
                                        work_dir=sfm_work_dir,
                                        start_image_id=sfm_start_image_id + 2,
                                        frame_window=incremental_ba_frame,
                                        end_image_id=sfm_end_image_id)
pnp_user_tagged_op_id = '620b6098423db05ad8d0027d'
n_view_sfm_forward_op.calculate_op_sfm_forward_incremental_ba(new_operation=True,
                                                              user_tagged_pnp_op_ids=[pnp_user_tagged_op_id],
                                                              )




