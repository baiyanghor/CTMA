
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/cam_track_service"
session_id = '61c12c6bb59e5e7b672ae3a0'

from SFMOperations.SFMDataTypes import SFMTrajectory
from SFMOperations.SFMMongodbInterface import MongodbInterface

db_ops = MongodbInterface(work_dir)
traj_list = db_ops.get_forward_trajectory_list(session_id, 5, 6)
print(f"{len(traj_list)} trajectories found")
for a_traj in traj_list:
    print(a_traj)
