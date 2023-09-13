from typing import List, Dict
import numpy as np
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.SFMCalculationNodes import CalNodeBase
from SFMOperations.SFMDataTypes import SFMTrajectory, SFMImage, SFM_OPERATION


class CensusTrajectories(CalNodeBase):
    def __init__(self, node_name='CensusTrajectories', session_id: str = '', work_dir='',
                 start_image_id: int = -1, end_image_id: int = -1, min_traj_length: int = -1):
        """

        :param node_name:
        :param session_id:
        :param work_dir:
        :param start_image_id: Real image ID
        :param end_image_id: Real image ID + 1
        :param min_traj_length:
        """
        super(CensusTrajectories, self).__init__(node_name, session_id, work_dir)
        if start_image_id >= end_image_id:
            raise ValueError("Check start image ID and end image ID range!")
        self._db_ops = MongodbInterface(work_dir)
        self._start_image_id = start_image_id
        self._end_image_id = end_image_id
        self._trajectories: Dict[int, SFMTrajectory] = {}
        self._sfm_image_list: List[SFMImage] = []
        self.dependent_nodes = ['CensusTrajectoryPairs']

    def _get_curr_section(self, section_image_id: int):
        """

        :param section_image_id:
        :return: return {trajectory_id: last_match_point}
        """
        if len(self._sfm_image_list) == 0:
            raise AttributeError("SFM image list not initialized yet!")
        if len(self._trajectories) == 0:
            return []

        section = {}
        for traj_id, sfm_traj in self._trajectories.items():
            if sfm_traj.get_next_image_id() == section_image_id:
                section.setdefault(traj_id, sfm_traj.corr_traj_pair[-1][1])
        return section

    def _register_trajectory_forward(self, traj_start_image_id: int, temp_traj_id: str, traj_pair: list):
        """
        Interst new trajectory record in database and return it's _id
        :param traj_start_image_id:
        :param temp_traj_id:
        :param match_point_idx:
        :return:
        """
        if len(self._sfm_image_list) == 0:
            raise IndexError("SFM image list not initialized yet!")
        new_trajectory = SFMTrajectory()
        new_trajectory.db_traj_id = temp_traj_id
        new_trajectory.start_image_id = traj_start_image_id
        new_trajectory.corr_traj_pair.append(traj_pair)
        new_trajectory.corr_image_points.append(
            self._sfm_image_list[traj_start_image_id].matches_to_next[traj_pair[1]])
        new_trajectory.traject_length = 1
        new_traj_id = len(self._trajectories)
        self._trajectories.setdefault(new_traj_id, new_trajectory)

    def _connect_current_match_point(self, traj_id: int, image_id: int, traj_pair: list):
        self._trajectories[traj_id].traject_length += 1
        self._trajectories[traj_id].corr_traj_pair.append(traj_pair)
        self._trajectories[traj_id].corr_image_points.append(
            self._sfm_image_list[image_id].matches_to_pre[traj_pair[0]])
        self._sfm_image_list[image_id].sfm_selected_traj_pairs.append(traj_pair)

    def census_trajectories_forward(self, new_operation: bool, sfm_operation_id=''):
        # Get sfm_images
        self._sfm_image_list = self._db_ops.get_sfm_image_list(self.session_id)

        assert len(self._sfm_image_list) > 0, "Acquire SFM image list failure!"

        # Get trajectory pairs
        traj_pairs_list = self._db_ops.get_trajectory_pairs(self.session_id)
        # Start from first images register trajectories with trajectory pairs
        temp_traj_id = ''
        for a_traj_pair in traj_pairs_list[self._start_image_id]:
            self._register_trajectory_forward(self._start_image_id, temp_traj_id, a_traj_pair)
            self._sfm_image_list[self._start_image_id].sfm_selected_traj_pairs.append(a_traj_pair)

        # Loop through sfm_images
        for image_idx in range(self._start_image_id + 1, self._end_image_id):
            # Generate section of existed trajectories
            section = self._get_curr_section(image_idx)
            section_to_previous = dict(sorted(section.items(), key=lambda kv: kv[0]))
            section_traj_id_list = list(section_to_previous.keys())
            section_point_id_list = np.array(list(section_to_previous.values()), dtype=int)
            # Loop through trajectory pairs in current image
            # Check current image match point with last trajectory section
            current_traj_pairs = np.array(traj_pairs_list[image_idx], dtype=int)
            connected_pids, idx_of_traj_pair_list, idx_of_pids_in_section =\
                np.intersect1d(current_traj_pairs[:, 0], section_point_id_list, return_indices=True)

            # print(f"Image ID {image_idx} with {len(connected_pids)} intersections")
            if len(connected_pids) > 0:
                # Add current image match point to trajectories in connected_pids
                for traj_pair_id, idx_in_section in zip(idx_of_traj_pair_list, idx_of_pids_in_section):
                    traj_id = section_traj_id_list[idx_in_section]
                    self._connect_current_match_point(traj_id, image_idx,
                                                      current_traj_pairs[traj_pair_id].tolist())

                rest_traj_pair_indices = list(set(range(current_traj_pairs.shape[0])) - set(idx_of_traj_pair_list))

                # Register new trajectory in the rest of match point in current image
                if current_traj_pairs[0, 1] != -1:
                    for traj_pair_idx in rest_traj_pair_indices:
                        self._register_trajectory_forward(image_idx, temp_traj_id, current_traj_pairs[traj_pair_idx].tolist())

            else:
                # If no new continuous in current image register new trajectory start from current image
                # else join image match point in current image to registered trajectory
                if current_traj_pairs[0, 1] != -1:
                    for a_traj_pair in current_traj_pairs:
                        self._register_trajectory_forward(image_idx, temp_traj_id, a_traj_pair.tolist())

        # Save trajectories to database
        if new_operation:
            sfm_operation_id = self._db_ops.new_sfm_operation(self.session_id, SFM_OPERATION.CENSUS_TRAJECTORIES,
                                                              list(range(self._start_image_id, self._end_image_id)))

        if len(self._trajectories) > 0:
            self._db_ops.save_census_trajectories(self.session_id, sfm_operation_id, self._trajectories)
        self._db_ops.set_sfm_selected_traj_pairs(self.session_id, sfm_operation_id, self._sfm_image_list)

    def clear_trajectories(self):
        """
        Clear trajectory with session id
        :return:
        """
        ...

