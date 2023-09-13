import os.path
import numpy as np
from json import dump
from scipy.spatial.transform import Rotation
from SFMOperations.SFMUtilities import get_ordered_image_ids_selection
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMConfigure.SFMConfigureOperators import SFMConfigureOps
from SFMOperations.SFMDataTypes import BA_PHASE, SFM_OPERATION


class MayaDataInterface(object):
    def __init__(self, session_id='', work_dir=''):
        self._session_id = session_id
        self.__work_dir = work_dir
        self._conf_ops = SFMConfigureOps(self.__work_dir)
        self._db_ops = MongodbInterface(self.__work_dir)

    def camera_rt_to_maya(self, rt: np.ndarray) -> (np.ndarray, np.ndarray):
        r_mat = rt[:, :3]
        tvec = rt[:, [3]]
        rotator_x_180 = Rotation.from_euler('x', 180, degrees=True)
        r_mat_x_180 = rotator_x_180.as_matrix()
        opengl_r_mat = np.matmul(r_mat.T, r_mat_x_180)
        rotator_gl = Rotation.from_matrix(opengl_r_mat)
        maya_rotation = rotator_gl.as_euler('xyz', degrees=True)
        cam_center = -np.matmul(r_mat.T, tvec)

        return maya_rotation.flatten(), cam_center.flatten()

    def write_rt_to_scene_file(self):
        user_cache_path = os.path.join(self._conf_ops.get_user_cache_path(), self._session_id)
        filename = self._conf_ops.get_camera_translation_filename()
        rt_to_scene_list = self._db_ops.get_camera_rt_to_scene_list(self._session_id)
        rts_to_write = []
        for row in rt_to_scene_list:
            m_r, m_c = self.camera_rt_to_maya(np.array(row['Rt_to_scene']))
            new_dict = {
                row['image_id']: {
                    "rotate": m_r.tolist(),
                    "translate": m_c.tolist()
                }
            }
            rts_to_write.append(new_dict)

        if not os.path.isdir(user_cache_path):
            os.mkdir(user_cache_path)

        full_filename = os.path.join(user_cache_path, filename)
        with open(full_filename, 'w') as file_handler:
            dump(rts_to_write, file_handler)
            return True

    def write_world_points_file(self):
        user_cache_path = os.path.join(self._conf_ops.get_user_cache_path(), self._session_id)
        filename = self._conf_ops.get_world_points_filename()
        world_points_data = self._db_ops.get_last_world_points(self._session_id, BA_PHASE.BA_GLOBAL_ITERATIVE)
        world_ponits_dict = {
            "world_points": world_points_data
        }

        if not os.path.isdir(user_cache_path):
            os.mkdir(user_cache_path)

        full_filename = os.path.join(user_cache_path, filename)
        with open(full_filename, 'w') as file_handler:
            dump(world_ponits_dict, file_handler)
            return True

    def write_op_world_points_file(self, sfm_operation_id: str):
        user_cache_path = os.path.join(self._conf_ops.get_user_cache_path(), self._session_id)
        filename = self._conf_ops.get_world_points_filename()
        world_points_data = self._db_ops.get_specify_last_world_points(self._session_id,
                                                                       sfm_operation_id,
                                                                       SFM_OPERATION.GLOBAL_BA_ITERATIVELY)
        world_ponits_dict = {
            "world_points": world_points_data
        }

        if not os.path.isdir(user_cache_path):
            os.mkdir(user_cache_path)

        full_filename = os.path.join(user_cache_path, filename)
        with open(full_filename, 'w') as file_handler:
            dump(world_ponits_dict, file_handler)
            return True

    def write_specify_rt_to_scene_file(self, ordered_merge_operation_ids: list, image_id_range: list):
        if len(image_id_range) == 0:
            raise ValueError("Empty image ID range")

        merged_image_id_dict = self._db_ops.get_op_affected_range_list(self._session_id, ordered_merge_operation_ids)

        image_id_selection_dict = get_ordered_image_ids_selection(image_id_range,
                                                                  ordered_merge_operation_ids,
                                                                  merged_image_id_dict)
        output_rt_list = {}
        for op_id in ordered_merge_operation_ids:
            operating_rt_list = self._db_ops.get_op_rt_to_scene_list(self._session_id,
                                                                     sfm_operation_id=op_id,
                                                                     image_id_range=image_id_selection_dict[op_id])
            output_rt_list |= operating_rt_list

        rts_to_write = []
        for image_id in image_id_range:
            m_r, m_c = self.camera_rt_to_maya(np.array(output_rt_list[str(image_id)]))
            new_dict = {
                str(image_id): {
                    "rotate": m_r.tolist(),
                    "translate": m_c.tolist()
                }
            }
            rts_to_write.append(new_dict)

        user_cache_path = os.path.join(self._conf_ops.get_user_cache_path(), self._session_id)
        filename = self._conf_ops.get_camera_translation_filename()
        if not os.path.isdir(user_cache_path):
            os.mkdir(user_cache_path)

        full_filename = os.path.join(user_cache_path, filename)
        with open(full_filename, 'w') as file_handler:
            dump(rts_to_write, file_handler)
            return True

