import os
import json


class SFMConfigureOps(object):
    def __init__(self, in_work_dir: str):
        self.configure_file = os.path.join(os.path.abspath(in_work_dir), "SFMConfigure/sfm-configure.json")
        self.configure_data = None
        if os.path.isfile(self.configure_file):
            with open(self.configure_file, 'r') as fileHandler:
                self.configure_data = json.load(fileHandler)
        else:
            print("Debug: Configure file does not exists!")

    def get_database_filename(self):
        if self.configure_data is not None:
            if 'sfm_database_filename' in self.configure_data:
                return self.configure_data['sfm_database_filename']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_mongodb_connect_string(self):
        if self.configure_data is not None:
            if 'sfm_mongodb_server' in self.configure_data and 'sfm_mongodb_port' in self.configure_data:
                return self.configure_data['sfm_mongodb_server'], self.configure_data['sfm_mongodb_port']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_sfm_db_name(self):
        if self.configure_data is not None:
            if 'sfm_mongodb_dbname' in self.configure_data:
                return self.configure_data['sfm_mongodb_dbname']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_sfm_global_info_coll_name(self):
        if self.configure_data is not None:
            if 'coll_sfm_global_info' in self.configure_data:
                return self.configure_data['coll_sfm_global_info']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_session_manage_coll_name(self):
        if self.configure_data is not None:
            if 'coll_session_manager' in self.configure_data:
                return self.configure_data['coll_session_manager']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_sfm_image_coll_name(self):
        if self.configure_data is not None:
            if 'coll_sfm_image' in self.configure_data:
                return self.configure_data['coll_sfm_image']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_user_image_points_coll_name(self):
        if self.configure_data is not None:
            if 'coll_user_image_points' in self.configure_data:
                return self.configure_data['coll_user_image_points']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_user_world_points_coll_name(self):
        if self.configure_data is not None:
            if 'coll_user_world_points' in self.configure_data:
                return self.configure_data['coll_user_world_points']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_sfm_camera_coll_name(self):
        if self.configure_data is not None:
            if 'coll_sfm_camera' in self.configure_data:
                return self.configure_data['coll_sfm_camera']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_sfm_operations_coll_name(self):
        if self.configure_data is not None:
            if 'coll_sfm_operations' in self.configure_data:
                return self.configure_data['coll_sfm_operations']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_operating_sfm_camera_coll_name(self):
        if self.configure_data is not None:
            if 'coll_operating_sfm_camera' in self.configure_data:
                return self.configure_data['coll_operating_sfm_camera']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_shadow_mask_path(self):
        if self.configure_data is not None:
            if 'sfm_shadow_mask_path' in self.configure_data:
                return self.configure_data['sfm_shadow_mask_path']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_valid_image_type(self):
        if self.configure_data is not None:
            if 'valid_image_type' in self.configure_data:
                return self.configure_data['valid_image_type']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_user_cache_path(self):
        if self.configure_data is not None:
            if 'user_cache_path' in self.configure_data:
                return self.configure_data['user_cache_path']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_camera_translation_filename(self):
        if self.configure_data is not None:
            if 'camera_translation_filename' in self.configure_data:
                return self.configure_data['camera_translation_filename']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_user_tagged_points_file(self):
        if self.configure_data is not None:
            if 'user_tagged_json_file' in self.configure_data:
                return self.configure_data['user_tagged_json_file']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_user_tag_points_coll_name(self):
        if self.configure_data is not None:
            if 'coll_user_tagged_points' in self.configure_data:
                return self.configure_data['coll_user_tagged_points']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_world_points_filename(self):
        if self.configure_data is not None:
            if 'world_points_file_name' in self.configure_data:
                return self.configure_data['world_points_file_name']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_sfm_forward_trajectory_coll_name(self):
        if self.configure_data is not None:
            if 'coll_forward_trajectories' in self.configure_data:
                return self.configure_data['coll_forward_trajectories']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_sfm_backward_trajectory_coll_name(self):
        if self.configure_data is not None:
            if 'coll_backward_trajectories' in self.configure_data:
                return self.configure_data['coll_backward_trajectories']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_op_ba_last_wp_coll_name(self):
        if self.configure_data is not None:
            if 'coll_op_last_world_pts' in self.configure_data:
                return self.configure_data['coll_op_last_world_pts']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_op_ba_benchmark_coll_name(self):
        if self.configure_data is not None:
            if 'coll_op_ba_benchmark' in self.configure_data:
                return self.configure_data['coll_op_ba_benchmark']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_op_ba_benchmark_per_image_coll_name(self):
        if self.configure_data is not None:
            if 'coll_op_ba_benchmark_per_image' in self.configure_data:
                return self.configure_data['coll_op_ba_benchmark_per_image']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_census_trajectories_coll_name(self) -> str:
        if self.configure_data is not None:
            if 'coll_sfm_census_trajectories' in self.configure_data:
                return self.configure_data['coll_sfm_census_trajectories']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_census_triangulation_coll_name(self):
        if self.configure_data is not None:
            if 'coll_sfm_census_triangulation' in self.configure_data:
                return self.configure_data['coll_sfm_census_triangulation']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_user_tagged_trajectories_coll_name(self):
        if self.configure_data is not None:
            if 'coll_user_tagged_trajectories' in self.configure_data:
                return self.configure_data['coll_user_tagged_trajectories']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None


