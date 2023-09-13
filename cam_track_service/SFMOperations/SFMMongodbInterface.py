import numpy as np
from typing import List, Dict
from datetime import datetime
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from SFMConfigure.SFMConfigureOperators import SFMConfigureOps
from SFMOperations.SFMDataTypes import SFMImage, MatchedImagePoint, \
    SFM_STATUS, SFMTrajectory, SFM_OPERATION, SFMWorldPoint


class MongodbInterface(object):

    def __init__(self, work_dir: str):
        self._config_ops = SFMConfigureOps(work_dir)
        mongo_server, mongo_port = self._config_ops.get_mongodb_connect_string()
        self.mongo_server = mongo_server
        self.mongo_port = mongo_port

    def new_session(self, user_name, user_location) -> str:
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_session_manager = db_manager[self._config_ops.get_session_manage_coll_name()]
            new_session_dict = {
                "user_name": user_name,
                "user_location": user_location,
                "session_time": datetime.utcnow(),
                "sfm_operations": [],
                "status": {
                    SFM_STATUS.MATCHED_FEATURE_SAVED: {"done": 0, "dependent": []},
                    SFM_STATUS.INITIAL_SFM_PAIR: {"done": 0,
                                                  "dependent": [SFM_STATUS.MATCHED_FEATURE_SAVED]},
                    SFM_STATUS.USER_TAGGED_FEATURE_DONE: {"done": 0, "dependent": []},
                    SFM_STATUS.FIRST_PNP_READY: {"done": 0,
                                                 "dependent": [SFM_STATUS.INITIAL_SFM_PAIR,
                                                               SFM_STATUS.USER_TAGGED_FEATURE_DONE]},
                    SFM_STATUS.INITIAL_SFM_ROUND_READY: {"done": 0,
                                                         "dependent": [SFM_STATUS.MATCHED_FEATURE_SAVED,
                                                                       SFM_STATUS.INITIAL_SFM_PAIR,
                                                                       SFM_STATUS.USER_TAGGED_FEATURE_DONE,
                                                                       SFM_STATUS.FIRST_PNP_READY]}
                },
                "last_done": ""
            }

            new_session_id_obj = coll_session_manager.insert_one(new_session_dict).inserted_id
            new_session_id = f"{new_session_id_obj}"

            return new_session_id

    def set_sfm_global_info(self, session_id: str, i_focal_length: str, i_image_count: str, i_image_size: str,
                            i_image_path: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_global_sfm = db_manager[self._config_ops.get_sfm_global_info_coll_name()]
            height, width = i_image_size.split(',')
            if coll_global_sfm.count_documents({"session_id": session_id}) == 0:

                global_sfm_dict = {
                    "session_id": session_id,
                    "focal_length": float(i_focal_length),
                    "image_count": int(i_image_count),
                    "image_size": {"height": int(height), "width": int(width)},
                    "images_path": i_image_path
                }
                coll_global_sfm.insert_one(global_sfm_dict)
            else:

                coll_global_sfm.update_one({"session_id": session_id},
                                           {
                                               "$set": {
                                                   "focal_length": float(i_focal_length),
                                                   "image_count": int(i_image_count),
                                                   "image_size.height": int(height),
                                                   "image_size.width": int(width),
                                                   "images_path": i_image_path
                                               }
                                           })

            return True

    def get_images_info(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_global_info = db_manager[self._config_ops.get_sfm_global_info_coll_name()]
            a_sfm_global_info = coll_sfm_global_info.find_one({"session_id": session_id})
            if isinstance(a_sfm_global_info, dict):
                size_tuple = (a_sfm_global_info["image_size"]["height"],
                              a_sfm_global_info["image_size"]["width"])

                return a_sfm_global_info["focal_length"], size_tuple, a_sfm_global_info["image_count"], \
                       a_sfm_global_info["images_path"]

        return None

    def set_session_status(self, session_id: str, new_status: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_session_manager = db_manager[self._config_ops.get_session_manage_coll_name()]
            coll_session_manager.update_one({"_id": ObjectId(session_id)},
                                            {"$set": {".".join(["status", new_status, "done"]): int(1),
                                                      "last_done": new_status}})
            return True

    def save_initial_sfm_pair(self, session_id: str, sfm_initial_pair: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_global_info = db_manager[self._config_ops.get_sfm_global_info_coll_name()]
            coll_sfm_global_info.update_one({'session_id': session_id},
                                            {"$set": {"sfm_initial_pair": sfm_initial_pair}})
            return True

    def clear_sfm_image_by_session(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            coll_sfm_image.delete_many({"session_id": session_id})

    def save_sfm_matched_features(self, session_id: str, memory_data: List[SFMImage]):
        assert len(memory_data) > 0
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_session_manager = db_manager[self._config_ops.get_session_manage_coll_name()]

            if coll_session_manager.count_documents({"_id": ObjectId(session_id)}) == 0:
                return False

            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]

            if coll_sfm_image.count_documents({"session_id": session_id}) > 0:
                coll_sfm_image.delete_many({"session_id": session_id})

            for a_sfm_image in memory_data:
                match_to_pre_list = {}
                match_to_next_list = {}
                if len(a_sfm_image.matches_to_pre) > 0:
                    for i, a_matched_point in enumerate(a_sfm_image.matches_to_pre):
                        match_to_pre_list.setdefault(str(i),
                                                     {"sfm_image_id": a_matched_point.sfm_image_id,
                                                      "world_point_id": a_matched_point.world_point_id,
                                                      "position_2d": a_matched_point.position_2d.tolist()})
                if len(a_sfm_image.matches_to_next) > 0:
                    for i, a_matched_point in enumerate(a_sfm_image.matches_to_next):
                        match_to_next_list.setdefault(str(i),
                                                      {"sfm_image_id": a_matched_point.sfm_image_id,
                                                       "world_point_id": a_matched_point.world_point_id,
                                                       "position_2d": a_matched_point.position_2d.tolist()})

                insert_dict = {
                    "session_id": session_id,
                    "image_id": a_sfm_image.image_id,
                    "image_file_name": a_sfm_image.image_file_name,
                    "match_to_pre_list": match_to_pre_list,
                    "match_to_next_list": match_to_next_list
                }

                coll_sfm_image.insert_one(insert_dict)

            return True

    def get_sfm_image_list(self, session_id: str):
        sfm_list = []
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            sfm_image_dicts = coll_sfm_image.find({"session_id": session_id}).sort('image_id', pymongo.ASCENDING)
            for a_sfm_image_dict in sfm_image_dicts:
                new_sfm_image = SFMImage()
                new_sfm_image.image_id = a_sfm_image_dict['image_id']
                new_sfm_image.matches_to_pre = []
                new_sfm_image.image_file_name = a_sfm_image_dict['image_file_name']
                # new_sfm_image.mask_file_name = a_sfm_image_dict['mask_file_name']
                match_to_pre_list = a_sfm_image_dict['match_to_pre_list']

                if len(match_to_pre_list) > 0:
                    for idx, data in sorted(match_to_pre_list.items(), key=lambda x: int(x[0])):
                        new_point = MatchedImagePoint()
                        new_point.point_id = int(idx)
                        new_point.sfm_image_id = data['sfm_image_id']
                        new_point.world_point_id = data['world_point_id']
                        new_point.position_2d = np.array(data['position_2d'])
                        new_sfm_image.matches_to_pre.append(new_point)
                else:
                    new_sfm_image.matches_to_pre = []

                new_sfm_image.matches_to_next = []
                match_to_next_list = a_sfm_image_dict['match_to_next_list']
                if len(match_to_next_list) > 0:
                    for idx, data in sorted(match_to_next_list.items(), key=lambda x: int(x[0])):
                        new_point = MatchedImagePoint()
                        new_point.point_id = int(idx)
                        new_point.sfm_image_id = data['sfm_image_id']
                        new_point.world_point_id = data['world_point_id']
                        new_point.position_2d = np.array(data['position_2d'])
                        new_sfm_image.matches_to_next.append(new_point)
                else:
                    new_sfm_image.matches_to_next = []

                sfm_list.append(new_sfm_image)

        return sfm_list

    def save_sfm_initial_image_pair_id(self, session_id: str, sfm_initial_pair: tuple):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_global_info = db_manager[self._config_ops.get_sfm_global_info_coll_name()]
            coll_sfm_global_info.update_one({"session_id": session_id},
                                            {"$set": {"sfm_initial_pair_id": list(sfm_initial_pair)}})

            return True

    def get_last_session_status(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_session_manager = db_manager[self._config_ops.get_session_manage_coll_name()]
            ret_data = coll_session_manager.find_one({"_id": ObjectId(session_id)})
            last_status = ret_data['last_done']

            return last_status

    def check_sfm_status_done(self, session_id: str, check_status: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_session_manager = db_manager[self._config_ops.get_session_manage_coll_name()]
            ret_data = coll_session_manager.find_one({"_id": ObjectId(session_id)})
            last_status = ret_data['status'][check_status]['done']
            if last_status == 1:
                return True

        return False

    def get_sfm_initial_pair_id(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_global_info = db_manager[self._config_ops.get_sfm_global_info_coll_name()]
            ret_data = coll_sfm_global_info.find_one({"session_id": session_id})
            sfm_initial_pair_id = ret_data['sfm_initial_pair_id']

            return sfm_initial_pair_id

    def set_user_selected_image_point(self, session_id: str, frame: int, c_id: int, x: float, y: float):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_user_image_points = db_manager[self._config_ops.get_user_image_points_coll_name()]

            filter_image_point_dict = {
                "$and": [
                    {"session_id": session_id},
                    {"frame": int(frame)},
                    {"corres_id": int(c_id)}
                ]
            }

            update_image_point_dict = {
                "$set": {
                    "session_id": session_id,
                    "frame": int(frame),
                    "corres_id": int(c_id),
                    "x": float(x),
                    "y": float(y)
                }
            }

            coll_user_image_points.update_one(filter=filter_image_point_dict, update=update_image_point_dict,
                                              upsert=True)

            return True

    def set_user_selected_world_point(self, session_id: str, c_id: int, x: float, y: float, z: float):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_user_world_points = db_manager[self._config_ops.get_user_world_points_coll_name()]

            filter_world_point_dict = {
                "$and": [
                    {"session_id": session_id},
                    {"corres_id": int(c_id)}
                ]
            }

            update_world_point_dict = {
                "$set": {
                    "session_id": session_id,
                    "corres_id": int(c_id),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                }
            }

            coll_user_world_points.update_one(filter=filter_world_point_dict, update=update_world_point_dict,
                                              upsert=True)

            return True

    def get_user_world_points(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_user_world_points = db_manager[self._config_ops.get_user_world_points_coll_name()]
            ret_data = coll_user_world_points.find({"session_id": session_id},
                                                   {"_id": False,
                                                    "x": True,
                                                    "y": True,
                                                    "z": True}).sort('corres_id', pymongo.ASCENDING)
            user_world_point_list = []
            for a_row in ret_data:
                user_world_point_list.append([a_row['x'], a_row['y'], a_row['z']])

            return np.array(user_world_point_list)

    def get_user_images_points(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_user_image_points = db_manager[self._config_ops.get_user_image_points_coll_name()]
            coll_sfm_global_info = db_manager[self._config_ops.get_sfm_global_info_coll_name()]
            ret_data = coll_sfm_global_info.find_one({"session_id": session_id},
                                                     {"_id": False, "sfm_initial_pair_id": True})
            pair_id = ret_data["sfm_initial_pair_id"]
            image_point_data_0 = coll_user_image_points.find(
                {"$and": [{"session_id": session_id}, {"frame": pair_id[0] + 1}]},
                {"_id": False, "x": True, "y": True}).sort('corres_id', pymongo.ASCENDING)

            points_0 = []
            for a_point in image_point_data_0:
                points_0.append([a_point['x'], a_point['y']])
            image_point_data_1 = coll_user_image_points.find(
                {"$and": [{"session_id": session_id}, {"frame": pair_id[1] + 1}]},
                {"_id": False, "x": True, "y": True}).sort('corres_id', pymongo.ASCENDING)

            points_1 = []
            for a_point in image_point_data_1:
                points_1.append([a_point['x'], a_point['y']])

            return np.array([points_0, points_1])

    def get_user_tagged_features(self, session_id: str, last_user_tag_operation=False, sfm_operation_id=''):

        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_user_tagged_points = db_manager[self._config_ops.get_user_tag_points_coll_name()]

            if last_user_tag_operation:
                coll_sfm_operations = db_manager[self._config_ops.get_sfm_operations_coll_name()]
                tag_operations = coll_sfm_operations.find({"operation_type": SFM_OPERATION.USER_TAG_POINTS},
                                                          {'_id': True}).sort("operation_time", pymongo.DESCENDING)
                last_tag_op_id = str(tag_operations[0]['_id'])
                operation_data = coll_user_tagged_points.find_one(
                    {"session_id": session_id, 'sfm_operation_id': last_tag_op_id})
            else:
                operation_data = coll_user_tagged_points.find_one({"session_id": session_id,
                                                                   "sfm_operation_id": sfm_operation_id})

            features_dict = operation_data['features']
            user_tagged_image_ids = sorted(operation_data['tagged_frames'], reverse=False)
            features_ordered = sorted(features_dict.items(), key=lambda a: int(a[0]))

            im_points_one = []
            im_points_two = []
            world_points = []
            for feature_id, feature_data in features_ordered:
                world_points.append(feature_data['world_point'])
                im_points_one.append(feature_data[str(user_tagged_image_ids[0])]['image_point'])
                im_points_two.append(feature_data[str(user_tagged_image_ids[1])]['image_point'])

            return user_tagged_image_ids, np.array([im_points_one, im_points_two]), np.array(world_points)

    def get_sfm_initial_pair_frame(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_global_info = db_manager[self._config_ops.get_sfm_global_info_coll_name()]
            ret_data = coll_sfm_global_info.find_one({"session_id": session_id},
                                                     {"_id": False, "sfm_initial_pair_id": True})
            sfm_initial_pair = ret_data['sfm_initial_pair_id']

            return [sfm_initial_pair[0] + 1, sfm_initial_pair[1] + 1]

    def set_camera_rt_to_scene(self, session_id: str, image_id: int, rt_to_scene: np.ndarray):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_camera = db_manager[self._config_ops.get_sfm_camera_coll_name()]
            update_filter_dict = {
                "$and": [
                    {"session_id": session_id},
                    {"image_id": image_id}
                ]
            }

            update_dict = {
                "$set": {
                    "image_id": image_id,
                    "Rt_to_scene": rt_to_scene.tolist()
                }
            }

            coll_sfm_camera.update_one(filter=update_filter_dict, update=update_dict, upsert=True)

            return True

    def set_operating_rt_to_scene(self, session_id: str, operation_id: str, image_id: int, rt_to_scene: np.ndarray):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_operating_sfm_camera = db_manager[self._config_ops.get_operating_sfm_camera_coll_name()]
            update_filter_dict = {
                "session_id": session_id,
                "sfm_operation_id": operation_id
            }

            update_dict = {
                "$set": {
                    ".".join(["Rt_to_scene", str(image_id)]): rt_to_scene.tolist()
                }
            }

            coll_operating_sfm_camera.update_one(filter=update_filter_dict, update=update_dict, upsert=True)
            return True

    def set_operating_rt_to_scene_list(self, session_id: str, sfm_operation_id: str, rt_to_scene_dataset: dict):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_operating_sfm_camera = db_manager[self._config_ops.get_operating_sfm_camera_coll_name()]
            update_filter_dict = {
                "session_id": session_id,
                "sfm_operation_id": sfm_operation_id
            }

            update_dict = {
                "$set": {
                    "Rt_to_scene": rt_to_scene_dataset
                }
            }
            coll_operating_sfm_camera.update_one(filter=update_filter_dict, update=update_dict, upsert=True)
            return True

    def new_sfm_operation(self, session_id: str, sfm_operation_type: str, affected_range: list):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_operation = db_manager[self._config_ops.get_sfm_operations_coll_name()]
            insert_dict = {
                "session_id": session_id,
                "operation_type": sfm_operation_type,
                "operation_time": datetime.utcnow(),
                "affected_range": affected_range
            }
            new_sfm_operation_id = coll_sfm_operation.insert_one(insert_dict).inserted_id
            coll_session_management = db_manager[self._config_ops.get_session_manage_coll_name()]
            filter_dict = {
                "_id": ObjectId(session_id)
            }
            update_dict = {
                "$addToSet": {
                    "sfm_operations": f"{new_sfm_operation_id}"
                }
            }
            coll_session_management.update_one(filter_dict, update_dict, upsert=True)

            return f"{new_sfm_operation_id}"

    def get_camera_rt_to_scene(self, session_id: str, image_id: int):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_camera = db_manager[self._config_ops.get_sfm_camera_coll_name()]
            ret_data = coll_sfm_camera.find_one({"$and": [{"session_id": session_id}, {"image_id": image_id}]})
            return ret_data["Rt_to_scene"]

    def get_camera_rt_to_scene_list(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_camera = db_manager[self._config_ops.get_sfm_camera_coll_name()]
            ret_data = coll_sfm_camera.find({"session_id": session_id}).sort('image_id', pymongo.ASCENDING)
            return list(ret_data)

    def get_op_rt_to_scene_list(self, session_id: str, sfm_operation_id: str, image_id_range: list):
        """
        Further modification pending
        :param session_id:
        :param sfm_operation_id:
        :param image_id_range:
        :return:
        """
        if image_id_range is None:
            image_id_range = []

        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_operating_sfm_camera = db_manager[self._config_ops.get_operating_sfm_camera_coll_name()]
            ret_data = coll_operating_sfm_camera.find_one({"session_id": session_id,
                                                           "sfm_operation_id": sfm_operation_id})

            rt_to_scene_dict = ret_data['Rt_to_scene']
            if len(image_id_range) > 0:
                selected_rt_to_scene = {k: rt_to_scene_dict.get(k, None) for k in map(str, image_id_range)}
                return selected_rt_to_scene
            else:
                return rt_to_scene_dict

    def set_sfm_image_trajectory_pair(self, session_id: str, sfm_image_list: List[SFMImage]):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            for a_sfm_image in sfm_image_list:
                image_id = a_sfm_image.image_id
                trajectory_pairs = a_sfm_image.trajectory_pair.tolist()
                filter_dict = {
                    "$and": [
                        {"session_id": session_id},
                        {"image_id": image_id}
                    ]
                }
                update_dict = {
                    "$set": {
                        "trajectory_pairs": trajectory_pairs
                    }
                }
                coll_sfm_image.update_one(filter_dict, update_dict, upsert=True)
            return True

    def get_trajectory_pairs(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            ret_data = coll_sfm_image.find({"session_id": session_id},
                                           {"_id": False, "image_id": True,
                                            "trajectory_pairs": True}).sort("image_id", pymongo.ASCENDING)
            trajectory_pairs_list = []
            for a_row in ret_data:
                trajectory_pairs_list.append(a_row['trajectory_pairs'])

            return trajectory_pairs_list

    def set_masked_feature_indices(self, session_id: str, image_id: int, keep_indices_to_pre: np.ndarray,
                                   keep_indices_to_next: np.ndarray):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            filter_dict = {
                "$and": [
                    {"session_id": session_id},
                    {"image_id": image_id}
                ]
            }
            if keep_indices_to_pre is None:
                to_keep_indices_to_pre = []
            else:
                to_keep_indices_to_pre = keep_indices_to_pre.tolist()

            if keep_indices_to_next is None:
                to_keep_indices_to_next = []
            else:
                to_keep_indices_to_next = keep_indices_to_next.tolist()

            update_dict = {
                "$set": {
                    "keep_indices_to_next_list": to_keep_indices_to_next,
                    "keep_indices_to_pre_list": to_keep_indices_to_pre
                }
            }

            coll_sfm_image.update_one(filter_dict, update_dict, upsert=True)

            return True

    def set_mask_file_name(self, session_id: str, image_id: int, mask_file_name: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            filter_dict = {
                "session_id": session_id,
                "image_id": image_id
            }
            update_dict = {
                "$set": {
                    "mask_file_name": mask_file_name
                }
            }
            coll_sfm_image.update_one(filter_dict, update_dict, upsert=True)

    def get_masked_features(self, session_id: str, image_id: int):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            ret_data = coll_sfm_image.find_one({"session_id": session_id, "image_id": image_id},
                                               {"keep_indices_to_pre_list": True,
                                                "keep_indices_to_next_list": True,
                                                "match_to_pre_list": True,
                                                "match_to_next_list": True})

            keep_indices_to_pre = ret_data['keep_indices_to_pre_list']
            keep_indices_to_next = ret_data['keep_indices_to_next_list']

            match_to_pre_dict = ret_data['match_to_pre_list']
            match_to_next_dict = ret_data['natch_to_next_list']

            match_to_pre = []
            match_to_next = []

            for idx in keep_indices_to_pre:
                match_to_pre.append(match_to_pre_dict[str(idx)]['position_2d'])

            for idx in keep_indices_to_next:
                match_to_next.append(match_to_next_dict[str(idx)]['position_2d'])

            return np.array(match_to_pre), np.array(match_to_next)

    def clear_forward_trajectories(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_forward_trajectory_coll_name()]
            coll_sfm_trajectory.delete_many({"session_id": session_id})
            return True

    def clear_backward_trajectories(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_backward_trajectory_coll_name()]
            coll_sfm_trajectory.delete_many({"session_id": session_id})
            return True

    def set_forward_trajectory(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_forward_trajectory_coll_name()]
            coll_sfm_trajectory.delete_many({"session_id": session_id})
            return True

    def set_backward_trajectory(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_backward_trajectory_coll_name()]
            coll_sfm_trajectory.delete_many({"session_id": session_id})
            return True

    def save_forward_trajectory_list(self, session_id: str, trajectory_list: List[SFMTrajectory]):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_forward_trajectory_coll_name()]
            if len(trajectory_list) > 0:
                for a_trajectory in trajectory_list:
                    image_point_dict = {}
                    for a_match_point in a_trajectory.corr_image_points:
                        image_point_dict.setdefault(str(a_match_point.sfm_image_id), a_match_point.position_2d.tolist())

                    insert_dict = {
                        "session_id": session_id,
                        "start_image_id": a_trajectory.start_image_id,
                        "trajectory_length": a_trajectory.traject_length,
                        "image_point": image_point_dict
                    }

                    coll_sfm_trajectory.insert_one(insert_dict)

    def save_backward_trajectory_list(self, session_id: str, trajectory_list: List[SFMTrajectory]):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_backward_trajectory_coll_name()]
            if len(trajectory_list) > 0:
                for a_trajectory in trajectory_list:
                    image_point_dict = {}
                    for a_match_point in a_trajectory.corr_image_points:
                        image_point_dict.setdefault(str(a_match_point.sfm_image_id), a_match_point.position_2d.tolist())

                    insert_dict = {
                        "session_id": session_id,
                        "start_image_id": a_trajectory.start_image_id,
                        "trajectory_length": a_trajectory.traject_length,
                        "image_point": image_point_dict
                    }

                    coll_sfm_trajectory.insert_one(insert_dict)

    def get_forward_trajectory_list(self, session_id: str, start_image_id: int, minimum_traj_length=0):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_forward_trajectory_coll_name()]
            trajectory_list = []
            filter_dict = {
                "session_id": session_id,
                "start_image_id": start_image_id,
                "trajectory_length": {"$gte": minimum_traj_length}
            }

            ret_data = coll_sfm_trajectory.find(filter_dict).sort("trajectory_length", pymongo.DESCENDING)

            for row in ret_data:
                new_trajectory = SFMTrajectory()
                new_trajectory.start_image_id = row['start_image_id']
                new_trajectory.traject_length = row['trajectory_length']
                image_point_list = sorted(row['image_point'].items(), key=lambda t: int(t[0]), reverse=False)
                for im_id, position in image_point_list:
                    new_matchpoint = MatchedImagePoint()
                    new_matchpoint.sfm_image_id = int(im_id)
                    new_matchpoint.position_2d = np.array(position)
                    new_trajectory.corr_image_points.append(new_matchpoint)

                trajectory_list.append(new_trajectory)

            return trajectory_list

    def get_backward_trajectory_list(self, session_id: str, start_image_id: int, minimum_traj_length=0):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_trajectory = db_manager[self._config_ops.get_sfm_backward_trajectory_coll_name()]
            trajectory_list = []
            filter_dict = {
                "session_id": session_id,
                "start_image_id": start_image_id,
                "trajectory_length": {"$gte": minimum_traj_length}
            }

            ret_data = coll_sfm_trajectory.find(filter_dict).sort("trajectory_length", pymongo.DESCENDING)

            for row in ret_data:
                new_trajectory = SFMTrajectory()
                new_trajectory.start_image_id = row['start_image_id']
                new_trajectory.traject_length = row['trajectory_length']
                image_point_list = sorted(row['image_point'].items(), key=lambda t: int(t[0]), reverse=True)
                for im_id, position in image_point_list:
                    new_matchpoint = MatchedImagePoint()
                    new_matchpoint.sfm_image_id = int(im_id)
                    new_matchpoint.position_2d = np.array(position)
                    new_trajectory.corr_image_points.append(new_matchpoint)

                trajectory_list.append(new_trajectory)

            return trajectory_list

    def save_world_points(self, session_id: str, ba_phase_name: str, world_point_list: np.ndarray):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_last_world_points = db_manager['BALastWorldPoints']
            data_dict = {
                "session_id": session_id,
                ba_phase_name: {
                    "world_points": world_point_list.tolist()
                }
            }
            coll_ba_last_world_points.insert_one(data_dict)

        return True

    def save_op_last_world_points(self, session_id: str, sfm_operation_id: str, ba_phase_name: str,
                                  world_point_list: np.ndarray):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_op_ba_last_world_points = db_manager[self._config_ops.get_op_ba_last_wp_coll_name()]
            data_dict = {
                "session_id": session_id,
                "sfm_operation_id": sfm_operation_id,
                ba_phase_name: {
                    "world_points": world_point_list.tolist()
                }
            }
            coll_op_ba_last_world_points.insert_one(data_dict)
        return True

    def save_user_tagged_points(self, session_id: str, sfm_operation_id: str, user_tagged_dict: dict):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_user_tagged_points = db_manager[self._config_ops.get_user_tag_points_coll_name()]
            assert session_id == user_tagged_dict['session_id']

            insert_dict = {
                "session_id": session_id,
                "sfm_operation_id": sfm_operation_id,
                "tagged_frames": user_tagged_dict['frames'],
                "features": user_tagged_dict['features']
            }

            coll_user_tagged_points.insert_one(insert_dict)

        return True

    def get_last_world_points(self, session_id: str, ba_phase_name: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_last_world_points = db_manager['BALastWorldPoints']
            world_points = coll_ba_last_world_points.find_one(
                {"session_id": session_id, ba_phase_name: {"$exists": True}})
            return world_points[ba_phase_name]['world_points']

    def get_specify_last_world_points(self, session_id: str, sfm_operation_id: str, ba_phase_name: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_op_ba_last_world_points = db_manager[self._config_ops.get_op_ba_last_wp_coll_name()]
            world_points = coll_op_ba_last_world_points.find_one({"session_id": session_id,
                                                                  "sfm_operation_id": sfm_operation_id,
                                                                  ba_phase_name: {"$exists": True}})
            return world_points[ba_phase_name]['world_points']

    def clear_last_world_points(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_last_world_points = db_manager['BALastWorldPoints']
            delete_result = coll_ba_last_world_points.delete_many({'session_id': session_id})
            return delete_result.deleted_count

    def clear_op_last_world_points(self, session_id: str, sfm_operation_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_op_ba_last_world_points = db_manager[self._config_ops.get_op_ba_last_wp_coll_name()]
            delete_result = coll_op_ba_last_world_points.delete_many({'session_id': session_id,
                                                                      "sfm_operation_id": sfm_operation_id})
            return delete_result.deleted_count

    def clear_ba_residual(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark = db_manager['BABenchmark']
            coll_ba_benchmark.delete_many({"session_id": session_id})
        return True

    def clear_op_ba_residual(self, session_id: str, sfm_operation_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_op_ba_benchmark = db_manager[self._config_ops.get_op_ba_benchmark_coll_name()]
            ret = coll_op_ba_benchmark.delete_many({"session_id": session_id, "sfm_operation_id": sfm_operation_id})
        return ret.deleted_count

    def clear_ba_phase_residual(self, session_id: str, ba_phase_name: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark = db_manager['BABenchmark']
            coll_ba_benchmark.delete_many({"session_id": session_id, ba_phase_name: {"$exists": True}})
        return True

    def save_ba_residual(self, session_id: str, ba_phase_name: str, ba_doc: dict):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark = db_manager['BABenchmark']
            insert_data = {
                "session_id": session_id,
                ba_phase_name: ba_doc
            }
            coll_ba_benchmark.insert_one(insert_data)
        return True

    def save_op_ba_residual(self, session_id: str, sfm_operation_id: str, ba_phase_name: str, ba_doc: dict):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark = db_manager[self._config_ops.get_op_ba_benchmark_coll_name()]
            insert_data = {
                "session_id": session_id,
                "sfm_operation_id": sfm_operation_id,
                ba_phase_name: ba_doc
            }
            coll_ba_benchmark.insert_one(insert_data)
        return True

    def get_ba_benchmark(self, session_id: str, ba_phase_name: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark = db_manager['BABenchmark']
            ret_data = coll_ba_benchmark.find_one({"session_id": session_id, ba_phase_name: {"$exists": True}})
            return ret_data[ba_phase_name]

    def clear_ba_residual_perimage(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark_per_image = db_manager['BABenchmarkPerImage']
            coll_ba_benchmark_per_image.delete_many({"session_id": session_id})

        return True

    def clear_op_ba_residual_perimage(self, session_id: str, sfm_session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_op_ba_benchmark_per_image = db_manager[self._config_ops.get_op_ba_benchmark_per_image_coll_name()]
            coll_op_ba_benchmark_per_image.delete_many({"session_id": session_id, "sfm_operation_id": sfm_session_id})

        return True

    def save_ba_residual_perimage(self, session_id: str, sfm_image_id: int, ba_phase_name: str, ba_phase_data: dict):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark_per_image = db_manager['BABenchmarkPerImage']

            filter_dict = {
                "session_id": session_id,
                "sfm_image_id": sfm_image_id
            }

            update_dict = {
                "$set": {
                    ba_phase_name: ba_phase_data
                }
            }

            coll_ba_benchmark_per_image.update_one(filter_dict, update_dict, upsert=True)

        return True

    def save_op_ba_residual_perimage(self,
                                     session_id: str,
                                     sfm_operation_id: str,
                                     ba_phase_name: str,
                                     ba_phase_data: dict):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_op_ba_benchmark_per_image = db_manager[self._config_ops.get_op_ba_benchmark_per_image_coll_name()]

            filter_dict = {
                "session_id": session_id,
                "sfm_operation_id": sfm_operation_id,
            }

            update_dict = {
                "$set": {
                    ba_phase_name: ba_phase_data
                }
            }

            coll_op_ba_benchmark_per_image.update_one(filter_dict, update_dict, upsert=True)
        return True

    def get_ba_residual_perimage(self, session_id: str, sfm_image_id: int, ba_phase_name: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ba_benchmark_per_image = db_manager['BABenchmarkPerImage']
            ret_data = coll_ba_benchmark_per_image.find_one({"session_id": session_id,
                                                             "sfm_image_id": sfm_image_id,
                                                             ba_phase_name: {"$exists": True}})
            return ret_data[ba_phase_name]

    def get_last_operation(self, session_id: str, operation_type: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_session_management = db_manager[self._config_ops.get_session_manage_coll_name()]
            coll_sfm_operations = db_manager[self._config_ops.get_sfm_operations_coll_name()]
            typed_operation_set = coll_sfm_operations.find({"operation_type": operation_type},
                                                           {"_id": True, "affect_range": True})
            typed_operation_list = []
            for row in typed_operation_set:
                typed_operation_list.append(f"{row['_id']}")

            session_data = coll_session_management.find_one({"_id": ObjectId(session_id)})

            operation_list = session_data['sfm_operations']

            ret_id = ''
            for op_id in reversed(operation_list):
                if op_id in typed_operation_list:
                    ret_id = op_id
                    break

            affected_range = []
            typed_operation_set.rewind()
            for op in typed_operation_set:
                if op["_id"] == ObjectId(ret_id):
                    affected_range = op['affect_range']
                    break
            return ret_id, affected_range

    def get_op_affected_range(self, session_id: str, sfm_operation_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_operations = db_manager[self._config_ops.get_sfm_operations_coll_name()]
            ret_data = coll_sfm_operations.find_one({"_id": ObjectId(sfm_operation_id)},
                                                    {"_id": False, "session_id": True, "affected_range": True})
            if ret_data['session_id'] != session_id:
                return None

            return ret_data['affected_range']

    def get_op_affected_range_list(self, session_id: str, sfm_operation_ids: list):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_operations = db_manager[self._config_ops.get_sfm_operations_coll_name()]
            op_list = []
            for op_id in sfm_operation_ids:
                op_list.append(ObjectId(op_id))
            op_dataset = coll_sfm_operations.find({"session_id": session_id, "_id": {"$in": op_list}},
                                                  {"affected_range": True})
            affected_range_dict = {}
            for a_row in op_dataset:
                affected_range_dict.setdefault(str(a_row['_id']), a_row['affected_range'])

            return affected_range_dict

    def get_op_rt_to_scene(self, session_id: str, sfm_operation_id: str, image_id: int):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_operating_sfm_camera = db_manager[self._config_ops.get_operating_sfm_camera_coll_name()]
            filter_dict = {
                "session_id": session_id,
                "sfm_operation_id": sfm_operation_id
            }
            project_dict = {
                "_id": False,
                "Rt_to_scene." + str(image_id): True
            }

            rt_to_scene = coll_operating_sfm_camera.find_one(filter_dict, project_dict)

            return rt_to_scene['Rt_to_scene'][str(image_id)]

    def save_census_trajectories(self, session_id: str, sfm_operation_id: str,
                                 trajectories_dict: Dict[int, SFMTrajectory]):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_census_trajectories = db_manager[self._config_ops.get_census_trajectories_coll_name()]
            for _, a_trajectory in trajectories_dict.items():
                insert_dict = {
                    "session_id": session_id,
                    "sfm_operation_id": sfm_operation_id,
                    "start_image_id": a_trajectory.start_image_id,
                    "trajectory_length": a_trajectory.traject_length,
                    "image_points": {str(matched_point.sfm_image_id): matched_point.position_2d.tolist()
                                     for matched_point in a_trajectory.corr_image_points}
                }
                coll_census_trajectories.insert_one(insert_dict)

    def save_user_tagged_trajectories(self, session_id: str, sfm_operation_id: str, trajectories: List[SFMTrajectory]):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_user_tagged_trajectories = db_manager[self._config_ops.get_user_tagged_trajectories_coll_name()]
            for a_traj in trajectories:
                insert_dict = {
                    "session_id": session_id,
                    "sfm_operation_id": sfm_operation_id,
                    "start_image_id": a_traj.start_image_id,
                    "trajectory_length": a_traj.traject_length,
                    "image_points": {str(matched_point.sfm_image_id): matched_point.position_2d.tolist()
                                     for matched_point in a_traj.corr_image_points},
                    "world_point": a_traj.world_point.position_3d.tolist()
                }

                coll_user_tagged_trajectories.insert_one(insert_dict)




    def set_sfm_selected_traj_pairs(self, session_id: str, sfm_operation_id: str, sfm_image_list: List[SFMImage]):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_sfm_image = db_manager[self._config_ops.get_sfm_image_coll_name()]
            for sfm_image in sfm_image_list:
                filter_dict = {
                    "session_id": session_id,
                    "image_id": sfm_image.image_id
                }
                update_dict = {
                    "$set": {
                        ".".join(["sfm_selected_traj_pairs", sfm_operation_id]): sfm_image.sfm_selected_traj_pairs
                    }
                }
                coll_sfm_image.update_one(filter_dict, update_dict, upsert=True)

    def validate_census_traj_exists(self, session_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_name_census_trajectory = self._config_ops.get_census_trajectories_coll_name()
            return coll_name_census_trajectory in db_manager.list_collection_names() and not 0 >= db_manager[
                coll_name_census_trajectory].count_documents({"session_id": session_id})

    def get_census_trajectory_list(self, session_id: str, census_traj_id: str, start_image_id: int, end_image_id: int,
                                   minimum_traj_length=0):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_census_trajectories = db_manager[self._config_ops.get_census_trajectories_coll_name()]
            trajectory_list = []
            filter_dict = {
                "session_id": session_id,
                "sfm_operation_id": census_traj_id,
                "$expr": {"$and": [{"$lte": [{"$add": ["$start_image_id", "$trajectory_length"]}, end_image_id + 1]},
                                   {"$gte": ["$trajectory_length", minimum_traj_length]},
                                   {"$gte": ["$start_image_id", start_image_id]}
                                   ]
                          }
            }

            ret_data = coll_census_trajectories.find(filter_dict).sort("start_image_id", pymongo.ASCENDING)

            for row in ret_data:
                new_trajectory = SFMTrajectory()
                new_trajectory.db_traj_id = f"{row['_id']}"
                new_trajectory.start_image_id = row['start_image_id']
                new_trajectory.traject_length = row['trajectory_length']
                image_point_list = sorted(row['image_points'].items(), key=lambda t: int(t[0]), reverse=False)
                for im_id, position in image_point_list:
                    new_match_point = MatchedImagePoint()
                    new_match_point.sfm_image_id = int(im_id)
                    new_match_point.position_2d = np.array(position)
                    new_trajectory.corr_image_points.append(new_match_point)

                trajectory_list.append(new_trajectory)

            return trajectory_list

    def get_ops_rt_to_scene(self, session_id: str, ordered_sfm_op_ids: List[str]) -> dict:
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_ops_rt_to_scene = db_manager[self._config_ops.get_operating_sfm_camera_coll_name()]

            filter_dict = {
                "session_id": session_id,
                "sfm_operation_id": {"$in": ordered_sfm_op_ids}
            }
            project_dict = {
                "sfm_operation_id": True,
                "Rt_to_scene": True
            }
            ret_cursor = coll_ops_rt_to_scene.find(filter_dict, project_dict)
            ops_rt_to_scene_dict = {r['sfm_operation_id']: r['Rt_to_scene'] for r in ret_cursor}

            return ops_rt_to_scene_dict

    def save_census_triangulation(self, session_id: str, census_traj_op_id: str, sfm_operation_id: str,
                                  affected_image_range: list, initial_rts_to_scene: dict, estimate_world_points: dict):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_census_triangulation = db_manager[self._config_ops.get_census_triangulation_coll_name()]
            insert_dict = {
                "session_id": session_id,
                "sfm_operation_id": sfm_operation_id,
                "census_traj_op_id": census_traj_op_id,
                "affected_image_range": affected_image_range,
                "initial_rts_to_scene": initial_rts_to_scene,
                "estimated_world_points": estimate_world_points
            }
            ret = coll_census_triangulation.insert_one(insert_dict)
            return ret.inserted_id

    def get_triangulated_trajectories(self, session_id: str, census_trajectory_op_id: str,
                                      census_triangulation_op_id: str):
        with MongoClient(self.mongo_server, self.mongo_port) as sfm_client:
            db_manager = sfm_client[self._config_ops.get_sfm_db_name()]
            coll_census_trajectory = db_manager[self._config_ops.get_census_trajectories_coll_name()]
            coll_census_triangulation = db_manager[self._config_ops.get_census_triangulation_coll_name()]

            sfm_triangulated_traj_list = []

            triangulate_dataset = coll_census_triangulation.find_one(
                {"session_id": session_id, "sfm_operation_id": census_triangulation_op_id})
            estimated_world_pt_dataset = triangulate_dataset['estimated_world_points']
            initial_rts_to_scene = triangulate_dataset['initial_rts_to_scene']
            used_traj_ids = list(estimated_world_pt_dataset.keys())

            traj_filter_dict = {
                "session_id": session_id,
                "sfm_operation_id": census_trajectory_op_id,
                "_id": {"$in": [ObjectId(traj_id) for traj_id in used_traj_ids]}
            }

            traj_project_dict = {
                "_id": {"$toString": "$_id"},
                "start_image_id": True,
                "trajectory_length": True,
                "image_points": True
            }

            trajectory_dataset = coll_census_trajectory.find(traj_filter_dict, traj_project_dict)
            validate_traj_set = {r['_id'] for r in trajectory_dataset}

            if not validate_traj_set == set(used_traj_ids):
                raise ValueError(f"Required trajectory ids don't match in calculated result"
                                 f"\nused_traj_ids: \n{used_traj_ids}"
                                 f"\nvalidate_traj_set: \n{validate_traj_set}\nused_traj_ids\n{used_traj_ids}")

            trajectory_dataset.rewind()

            for a_traj in trajectory_dataset:
                new_traj = SFMTrajectory()
                new_traj.db_traj_id = a_traj['_id']
                new_traj.start_image_id = a_traj['start_image_id']
                new_traj.traject_length = a_traj['trajectory_length']
                new_world_point = SFMWorldPoint()

                for im_id, point_2d in a_traj['image_points'].items():
                    new_match_point = MatchedImagePoint()
                    new_match_point.sfm_image_id = int(im_id)
                    new_match_point.position_2d = np.array(point_2d)
                    new_world_point.camera_ids.append(int(im_id))
                    new_traj.corr_image_points.append(new_match_point)

                new_world_point.position_3d = np.array(estimated_world_pt_dataset[new_traj.db_traj_id])

                new_traj.world_point = new_world_point

                sfm_triangulated_traj_list.append(new_traj)

            return sfm_triangulated_traj_list, initial_rts_to_scene
