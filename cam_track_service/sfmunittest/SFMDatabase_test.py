# from SFMConfigure import SFMConfigureOperators
# from SFMOperations.SFMDatabaseInterface import SFMDatabaseInterface
from SFMOperations.SFMCalculationNodes import CensusTrajectoryPairs
import numpy as np

from SFMOperations.MayaDataInterface import MayaDataInterface
from SFMOperations.SFMMongodbInterface import MongodbInterface

work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py39_for_dist"
# sfmcfg_ops = SFMConfigureOperators.SFMConfigureOps(work_dir)
# sfm_database_inter = SFMDatabaseInterface(work_dir)
# sfm_database_inter.initial_sfm_database()

# sfm_database_inter.update_sfm_initial_pair((2,3))


# sfm_ops = InitialPNPExtrinsic(session_id="6194d494a834ebdb9f929d94", work_dir=work_dir)
#
# sfm_ops.calculate_initial_pnp()
#
# sfm_db_ops = MongodbInterface(work_dir)
# maya_data_interface = MayaDataInterface()
#
# Rt_to_scene = sfm_db_ops.get_camera_Rt_to_scene("6194d494a834ebdb9f929d94", 1)
#
# rotation, cam_center = maya_data_interface.camera_Rt_to_maya(np.array(Rt_to_scene))
# print(f"Rotation: \n{rotation}")
# print(f"Camera Center: \n {cam_center}")

cal_trajectory_pairs = CensusTrajectoryPairs(session_id="6196256f3c7525995be3f86e", work_dir=work_dir)
cal_trajectory_pairs.calculate_trajectories()


