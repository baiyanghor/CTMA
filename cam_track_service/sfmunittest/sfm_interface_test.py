import sys
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/cam_track_service"
from SFMOperations.SFMPipelineInterface import SFMPipelineInterface

sfm_ops = SFMPipelineInterface(work_dir)

sfm_ops.clear_user_cache()
