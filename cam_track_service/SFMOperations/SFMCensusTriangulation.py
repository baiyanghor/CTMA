import numpy as np
from typing import List
from collections import OrderedDict
from SFMOperations.SFMDataTypes import SFMImage, SFM_OPERATION

from SFMOperations.SFMCalculationNodes import CalNodeBase
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.SFMUtilities import ordered_merge_ops_rt_to_scene, intrinsic_from_image_info
from SFMOperations.VGGFunctions import X_from_xP_nonlinear


class CensusTriangulation(CalNodeBase):
    def __init__(self, node_name='CensusTriangulation', session_id: str = '', work_dir='',
                 start_image_id: int = -1, end_image_id: int = -1, min_traj_length: int = -1):
        super(CensusTriangulation, self).__init__(node_name, session_id, work_dir)
        if start_image_id >= end_image_id:
            raise ValueError("Check start image ID and end image ID range!")
        if min_traj_length < 1:
            raise ValueError("Proper minimum trajectory length required")

        self._db_ops = MongodbInterface(work_dir)
        if not self._db_ops.validate_census_traj_exists(self.session_id):
            raise Exception("Trajectories data is not exists!")

        self._start_image_id = start_image_id
        self._end_image_id = end_image_id
        self._min_traj_length = min_traj_length
        self._sfm_image_list: List[SFMImage] = []

    def census_triangulation_forward(self, new_operation: bool, census_traj_op_id: str, ordered_sfm_op_ids: List[str],
                                     sfm_operation_id=''):
        self._sfm_image_list = self._db_ops.get_sfm_image_list(self.session_id)
        assert len(self._sfm_image_list) > 0, "Acquire SFM image list failure!"
        cal_image_id_range = list(range(self._start_image_id, self._end_image_id + 1))
        if new_operation:
            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id, SFM_OPERATION.CENSUS_TRIANGULATION,
                                                              cal_image_id_range)
        traj_list = self._db_ops.get_census_trajectory_list(self.session_id, census_traj_op_id, self._start_image_id,
                                                            self._end_image_id,
                                                            self._min_traj_length)

        ops_rt_to_scene_dict = self._db_ops.get_ops_rt_to_scene(self.session_id, ordered_sfm_op_ids)
        select_image_id_range = list(range(self._start_image_id, self._end_image_id + 1))
        selected_rt_to_scene_dict = ordered_merge_ops_rt_to_scene(ops_rt_to_scene_dict, ordered_sfm_op_ids,
                                                                  select_image_id_range)

        rt_mat_list = OrderedDict()
        for op_id, image_id_range in selected_rt_to_scene_dict.items():
            op_rt_scene_section = self._db_ops.get_op_rt_to_scene_list(self.session_id, op_id, image_id_range)
            rt_mat_list |= op_rt_scene_section

        assert set(map(int, rt_mat_list.keys())) == set(cal_image_id_range), "Required image range is not in database!"

        focal_length, image_size, _, _ = self._db_ops.get_images_info(self.session_id)
        camera_mat = intrinsic_from_image_info(focal_length, image_size)
        triangulate_dataset = OrderedDict()
        for a_traj in traj_list:
            project_matrices = np.zeros((a_traj.traject_length, 3, 4))
            image_points = np.zeros((a_traj.traject_length, 2))
            for order, a_point in enumerate(a_traj.corr_image_points):
                image_id = a_point.sfm_image_id
                rt_to_scene = rt_mat_list[str(image_id)]
                project_matrices[order] = np.matmul(camera_mat, rt_to_scene)
                image_points[order] = a_point.position_2d
            homo_world_point = X_from_xP_nonlinear(pts_position_2d=image_points, project_matrices=project_matrices,
                                                   image_sizes=image_size)

            triangulate_dataset[a_traj.db_traj_id] = homo_world_point.flatten()[:3].tolist()

        self._db_ops.save_census_triangulation(self.session_id, census_traj_op_id, sfm_operation_id,
                                               select_image_id_range, rt_mat_list,
                                               triangulate_dataset)
