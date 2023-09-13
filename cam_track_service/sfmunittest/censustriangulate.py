from pprint import pprint as pp

import numpy as np

from SFMOperations.SFMCensusTriangulation import CensusTriangulation

sfm_work_dir = f"E:/VHQ/camera-track-study/track-with-model/CameraTrackModel" \
               f"Alignment_py310_operationally/cam_track_service"

session_id = '61ee21c8d6a0b5298d93d145'

to_merge_op_ids = ["61f24712ec6a5916e26cf19d", "61f24ab6447717201b1cb552", "61efd53d7eb5cd27d1c7b7b8"]

census_traj_op_id = "61f5f6ed53a164075ceb9fb2"

census_triangulate_op = CensusTriangulation(session_id=session_id, work_dir=sfm_work_dir, start_image_id=1,
                                            end_image_id=20, min_traj_length=6)

census_triangulate_op.census_triangulation_forward(new_operation=True,
                                                   census_traj_op_id=census_traj_op_id,
                                                   ordered_sfm_op_ids=to_merge_op_ids)
