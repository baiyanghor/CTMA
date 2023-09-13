import importlib
import sys
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/MayaInterface"
if work_dir not in sys.path:
    sys.path.append(work_dir)
    
import SFMCalculateInterface as sfm_cal_interface

sfm_cal_ops = sfm_cal_interface.SFMCalculatorInterface(work_dir)
# focal_length = '933.3352'
# image_size = '540,960'
# image_file_path = 'E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py39_for_dist/UnitTest/openCVAPITest/test-image'
new_session_id = sfm_cal_ops.new_session('HOR', '_')
# print new_session_id

# new_session_id = "618e131518407f116e180b87"
# ret = sfm_cal_ops.set_global_sfm_info(new_session_id, focal_length, '17', image_size, image_file_path)
# print ret

# sfm_cal_ops.get_initial_sfm_pair('618e131518407f116e180b87')

