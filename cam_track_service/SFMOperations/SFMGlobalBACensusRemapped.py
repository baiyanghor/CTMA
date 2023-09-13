import numpy as np
from cv2 import Rodrigues
from SFMOperations.SFMCalculationNodes import CalNodeBase
from SFMOperations.SFMBundleAdjustment import BundleAdjustment
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.SFMDataTypes import SFM_OPERATION
from SFMOperations.SFMUtilities import intrinsic_from_image_info, rt_to_scene_from_vec
from SFMOperations.VGGFunctions import X_from_xP_nonlinear
from collections import OrderedDict


class GlobalBACensusRemapped(CalNodeBase):
    def __init__(self, node_name='GlobalBACensusRemapped', session_id: str = '', work_dir='', start_image_id=-1,
                 end_image_id=-1):
        super(GlobalBACensusRemapped, self).__init__(node_name, session_id, work_dir)
        self._start_image_id = start_image_id
        self._end_image_id = end_image_id
        self._db_ops = MongodbInterface(self.work_dir)

    def calculate_remapped_ba(self, census_triangulation_op_ids: list, census_trajectory_op_id: str,
                              anchor_image_ids: list, new_operation: bool, sfm_operation_id='', iterate_num=1, ):
        triangulated_trajectories, initial_rts_to_scene = \
            self._db_ops.get_triangulated_trajectories(self.session_id, census_trajectory_op_id,
                                                       census_triangulation_op_ids[0])

        affected_image_range = list(range(self._start_image_id, self._end_image_id + 1))

        if new_operation:
            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id, SFM_OPERATION.GLOBAL_BA_CENSUS_REMAPPED,
                                                              affected_image_range)

        focal_length, image_size, _, _ = self._db_ops.get_images_info(self.session_id)
        camera_intrinsic = intrinsic_from_image_info(focal_length, image_size)

        ordered_rt_to_scene = sorted(initial_rts_to_scene.items(), key=lambda element: int(element[0]), reverse=False)

        camera_parameter_dict = OrderedDict()
        for image_id, rt_to_scene in ordered_rt_to_scene:
            nrt_to_scene = np.array(rt_to_scene)
            rvec = Rodrigues(nrt_to_scene[:, :3])[0].ravel()
            tvec = nrt_to_scene[:, 3].ravel()
            camera_parameter_dict[image_id] = np.hstack((rvec, tvec))

        temp_camera_parameter_ids = list(camera_parameter_dict.keys())

        if not set(map(str,affected_image_range)) == set(temp_camera_parameter_ids):
            print(f"affected_image_range:\n{affected_image_range}"
                  f"\ntemp_camera_parameter_ids:\n{temp_camera_parameter_ids}")
            raise ValueError("Required image id range not match with database!")

        world_point_list = []
        image_point_list = []
        id_map_camera_parameter_world_point_image_point = []
        world_point_idx = 0
        image_point_idx = 0
        for a_traj in triangulated_trajectories:
            world_point_list.append(a_traj.world_point.position_3d)
            for a_match_point in a_traj.corr_image_points:
                image_point_list.append(a_match_point.position_2d)
                id_map = [temp_camera_parameter_ids.index(str(a_match_point.sfm_image_id)), world_point_idx,
                          image_point_idx]
                id_map_camera_parameter_world_point_image_point.append(id_map)
                image_point_idx += 1

            world_point_idx += 1

        camera_parameter_list = np.array(list(camera_parameter_dict.values()))
        world_point_list = np.array(world_point_list)
        image_point_list = np.array(image_point_list)
        id_map_camera_parameter_world_point_image_point = np.array(id_map_camera_parameter_world_point_image_point)

        ba_ops = BundleAdjustment()
        ba_count = 0
        while ba_count < iterate_num:
            ba_count += 1
            cam_rvec_list, cam_tvec_list, ba_world_pt_list = \
                ba_ops.n_view_ba_sparsity_weighted_data_remapped(camera_intrinsic,
                                                                 camera_parameter_list,
                                                                 world_point_list,
                                                                 image_point_list,
                                                                 id_map_camera_parameter_world_point_image_point
                                                                 )
            for data_id, image_id in enumerate(temp_camera_parameter_ids):
                if image_id not in anchor_image_ids:
                    camera_parameter_list[data_id] = np.hstack((cam_rvec_list[data_id], cam_tvec_list[data_id]))

            h_world_point_list = np.empty((world_point_list.shape[0], 4, 1))
            for traj_id, a_traj in enumerate(triangulated_trajectories):
                project_matrices = np.empty((a_traj.traject_length, 3, 4))
                image_pt_list = np.empty((a_traj.traject_length, 2))
                for idx, a_match_point in enumerate(a_traj.corr_image_points):
                    image_id = a_match_point.sfm_image_id
                    image_rt_to_scene_idx = temp_camera_parameter_ids.index(str(image_id))
                    rt_to_scene = rt_to_scene_from_vec(camera_parameter_list[image_rt_to_scene_idx, :3],
                                                       camera_parameter_list[image_rt_to_scene_idx, 3:])
                    project_matrices[idx] = np.matmul(camera_intrinsic, rt_to_scene)
                    image_pt_list[idx] = a_match_point.position_2d
                h_world_point_list[traj_id] = X_from_xP_nonlinear(image_pt_list, project_matrices, image_size)

            world_point_list = np.squeeze(h_world_point_list[:, :3, :], axis=2)

            ba_mean, ba_stdvar = ba_ops.n_view_data_remapped_residual(camera_parameter_list,
                                                                      world_point_list,
                                                                      image_point_list,
                                                                      camera_intrinsic,
                                                                      id_map_camera_parameter_world_point_image_point)

            print(f"Bundle adjustment iter {ba_count} BA mean {ba_mean} , standard variation {ba_stdvar}")

        rt_to_scene_dataset = {}
        shifted_indices = range(len(temp_camera_parameter_ids))
        for result_idx, image_idx in zip(shifted_indices, affected_image_range):
            rt_to_scene = rt_to_scene_from_vec(cam_rvec_list[result_idx], cam_tvec_list[result_idx])
            rt_to_scene_dataset[str(image_idx)] = rt_to_scene.tolist()
        self._db_ops.set_operating_rt_to_scene_list(self.session_id, sfm_operation_id, rt_to_scene_dataset)

