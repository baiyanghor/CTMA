import sys
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_operationally/MayaInterface"
if not work_dir in sys.path:
    sys.path.append(work_dir)
    
import SFMDataLoader
reload(SFMDataLoader)
session_id = '61ee21c8d6a0b5298d93d145'
drawer_ops = SFMDataLoader.SFMDataLoader(session_id, work_dir, 'SFMCamera', 'sfm_', 1)
drawer_ops.drawSFMCameras()
# drawer_ops.drawWorldPoints()
