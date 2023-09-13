from SFMOperations.SFMCalculationNodes import TrajectoryForward
from SFMOperations.SFMCensusTrajectories import CensusTrajectories
sfm_work_dir = f"E:/VHQ/camera-track-study/track-with-model/" \
               f"CameraTrackModelAlignment_py310_operationally/cam_track_service"

session_id = '620b561a22e6af9a60695b81'
# census_operation_id = '61f5f6ed53a164075ceb9fb2'
#
# traj_forward_ops = TrajectoryForward(session_id=session_id, work_dir=sfm_work_dir, start_image_id=0, end_image_id=24)
#
# traj_forward_ops.calculate_trajectory_forward()

census_traj_ops = CensusTrajectories(session_id=session_id, work_dir=sfm_work_dir, start_image_id=0, end_image_id=24)

census_traj_ops.census_trajectories_forward(new_operation=True)

