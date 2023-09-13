import sys
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/MayaInterface"
if work_dir not in sys.path:
    sys.path.append(work_dir)

import MayaControl
reload(MayaControl)
mayaInterface = MayaControl.MayaInterfaceController()
mayaInterface.exe()
