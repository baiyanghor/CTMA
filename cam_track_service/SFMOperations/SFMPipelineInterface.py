import os
import shutil
from json import load
from SFMOperations.SFMDataTypes import SFMSessionType, SFM_STATUS, SFM_OPERATION
from SFMOperations.SFMCalculationNodes import ImageMatchedFeatures, SFMInitialPair, InitialPNPExtrinsic
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMConfigure.SFMConfigureOperators import SFMConfigureOps


class SFMPipelineInterface(object):

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.session_type = SFMSessionType.DATABASE
        self._conf_ops = SFMConfigureOps(self.work_dir)

    def new_session(self, user_name: str, user_location: str):
        sfm_db_ops = MongodbInterface(self.work_dir)
        session_id = sfm_db_ops.new_session(user_name, user_location)
        return session_id

    def get_initial_sfm_pair(self, session_id: str):
        sfm_db_ops = MongodbInterface(self.work_dir)
        if sfm_db_ops.check_sfm_status_done(session_id, SFM_STATUS.INITIAL_SFM_PAIR):
            return sfm_db_ops.get_sfm_initial_pair_id(session_id)

        sfm_feature_extractor = ImageMatchedFeatures(work_dir=self.work_dir, session_id=session_id)
        sfm_feature_extractor.set_image_list()
        sfm_feature_extractor.calculate_matched_features()
        sfm_initial_pair_cal = SFMInitialPair(session_id, work_dir=self.work_dir)
        sfm_initial_pair = sfm_initial_pair_cal.calculate_sfm_initial_pair()

        return sfm_initial_pair

    def set_sfm_global_info(self, session_id: str, focal_length: str, image_count: str, image_size: str,
                            image_path: str):
        sfm_db_ops = MongodbInterface(self.work_dir)
        return sfm_db_ops.set_sfm_global_info(session_id, focal_length, image_count, image_size, image_path)

    def get_last_session_status(self, session_id: str):
        sfm_db_ops = MongodbInterface(self.work_dir)
        return sfm_db_ops.get_last_session_status(session_id)

    def set_user_selected_image_point(self, session_id: str, frame: int, c_id: int, x: float, y: float):
        sfm_db_ops = MongodbInterface(self.work_dir)
        if sfm_db_ops.set_user_selected_image_point(session_id, frame, c_id, x, y):
            return True

        return False

    def set_user_selected_world_point(self, session_id: str, c_id: int, w_x: float, w_y: float, w_z: float):
        sfm_db_ops = MongodbInterface(self.work_dir)
        if sfm_db_ops.set_user_selected_world_point(session_id, c_id, w_x, w_y, w_z):
            return True

        return False

    def set_user_tagged_done(self, session_id: str, op_id: str):
        sfm_db_ops = MongodbInterface(self.work_dir)
        if op_id == '0':
            """New tagged operation"""
        # Check the temp json file exits
        json_file_name = os.path.join(self._conf_ops.get_user_cache_path(),
                                      session_id,
                                      self._conf_ops.get_user_tagged_points_file())
        # Load json file to database
        with open(json_file_name, 'r') as rfp:
            user_operation_data = load(rfp)
            frames = user_operation_data['frames']
            image_id_range = [frame for frame in frames]
            new_sfm_operation_id = sfm_db_ops.new_sfm_operation(session_id=session_id,
                                                                sfm_operation_type=SFM_OPERATION.USER_TAG_POINTS,
                                                                affected_range=image_id_range)
            sfm_db_ops.save_user_tagged_points(session_id, new_sfm_operation_id, user_operation_data)
        # Set database status
        return new_sfm_operation_id
        # return sfm_db_ops.set_session_status(session_id, SFM_STATUS.USER_TAGGED_FEATURE_DONE)
        # return operation ID

    def initial_pnp_extrinsic(self, session_id: str):
        initial_pnp_ops = InitialPNPExtrinsic(session_id=session_id, work_dir=self.work_dir)
        if initial_pnp_ops.calculate_initial_pnp():
            return True
        else:
            return False

    def clear_user_cache(self, session_id: str):
        user_cache_path = self._conf_ops.get_user_cache_path()
        session_user_cache_path = os.path.join(user_cache_path, session_id)
        if os.path.exists(session_user_cache_path):
            shutil.rmtree(session_user_cache_path)
            os.mkdir(session_user_cache_path)
