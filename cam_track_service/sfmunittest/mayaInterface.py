from SFMOperations.MayaDataInterface import MayaDataInterface
from SFMOperations.SFMPipelineInterface import SFMPipelineInterface

work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_operationally/cam_track_service"
session_id = '61ee21c8d6a0b5298d93d145'
merge_op_ids = ["620b532ecc205f0140fb92c3"]
image_id_range = list(range(1, 21))

# sfm_ops = SFMPipelineInterface(work_dir)
# sfm_ops.clear_user_cache(session_id=session_id)

maya_interface_ops = MayaDataInterface(session_id, work_dir)
# maya_interface_ops.write_Rt_to_scene_file()
# maya_interface_ops.write_world_points_file()

maya_interface_ops.write_specify_rt_to_scene_file(ordered_merge_operation_ids=merge_op_ids,
                                                  image_id_range=image_id_range)
# maya_interface_ops.write_op_world_points_file("61efd53d7eb5cd27d1c7b7b8")

