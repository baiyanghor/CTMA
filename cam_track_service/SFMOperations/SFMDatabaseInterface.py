import sqlite3 as sql3
import os
import numpy as np
from SFMConfigure.SFMConfigureOperators import SFMConfigureOps
from SFMOperations.SFMDataTypes import GroundTrue, GroundTrueList
from typing import List


class SFMDatabaseInterface(object):
    def __init__(self, in_work_dir: str):
        self.maya_data_base = None
        self.work_dir = in_work_dir
        self.configOps = SFMConfigureOps(self.work_dir)
        self.__database_conn = None
        self.sfm_initial_pair_TN = 'sfm_initial_pair'
        self.sfm_images_TN = 'sfm_images'
        self.session_management_TN = 'session_management'
        self.__user_selected_point_TN = 'user_selected_points'
        self.__client_db_conn = None
        self.__client_sql_command = None
        self.__InCleintImagesPoints_TN = 'TagPointsInImages'
        self.__InClientScenePoints_TN = 'TagPointsInScene'
        self.__sql_command = None
        self.__database_conn = None

    def initial_sfm_database(self):
        database_filename = os.path.join(self.work_dir, 'SFMDatabase/' + self.configOps.get_database_filename())
        # if os.path.isfile(database_filename):
        #     os.remove(database_filename)
        self.__database_conn = sql3.connect(database_filename)
        self.__sql_command = self.__database_conn.cursor()

        create_initial_pair_table_str = f"CREATE TABLE IF NOT EXISTS {self.sfm_initial_pair_TN}("\
                                        f"ID INTEGER PRIMARY KEY AUTOINCREMENT," \
                                        f"session_id TEXT," \
                                        f"first_image INTEGER," \
                                        f"second_image INTEGER" \
                                        f")"

        create_session_management_table_str = f"CREATE TABLE IF NOT EXISTS {self.session_management_TN}(" \
                                              f"ID INTEGER PRIMARY KEY AUTOINCREMENT, " \
                                              f"user TEXT, " \
                                              f"user_location TEXT, " \
                                              f"images_location TEXT, " \
                                              f"client_db_location TEXT, " \
                                              f"status_new_session INTEGER DEFAULT 0, " \
                                              f"status_initial_sfm_pair_ready INTEGER DEFAULT 0, " \
                                              f"status_image_load INTEGER DEFAULT 0" \
                                              f")"

        create_user_selected_points_str = f"CREATE TABLE IF NOT EXISTS {self.__user_selected_point_TN}(" \
                                          f"ID INTEGER PRIMARY KEY AUTOINCREMENT, " \
                                          f"session_id INTEGER, " \
                                          f"point_id INTEGER, " \
                                          f"world_x REAL, " \
                                          f"world_y REAL, " \
                                          f"world_z REAL, " \
                                          f"image_one_name TEXT, " \
                                          f"image_one_x REAL, " \
                                          f"image_one_y REAL, " \
                                          f"image_two_name TEXT, " \
                                          f"image_two_x REAL, " \
                                          f"image_two_y REAL, " \
                                          f"image_three_name TEXT, " \
                                          f"image_three_x REAL, " \
                                          f"image_three_y REAL, " \
                                          f"image_four_name TEXT, " \
                                          f"image_four_x REAL, " \
                                          f"image_four_y REAL)"

        self.__sql_command.execute(create_initial_pair_table_str)
        self.__sql_command.execute(create_session_management_table_str)
        self.__sql_command.execute(create_user_selected_points_str)
        self.__database_conn.commit()

    def update_sfm_initial_pair(self, image_pair: tuple, in_session_id):
        if self.__sql_command is None:
            print("Database not initialized yet!")
            return
        check_pair_exists_str = f"SELECT EXISTS (SELECT ID FROM {self.sfm_initial_pair_TN} " \
                                f"WHERE session_id='{in_session_id}')"
        self.__sql_command.execute(check_pair_exists_str)
        pair_exists = self.__sql_command.fetchone()
        if pair_exists == 1:
            update_initial_image_pair_str = f"UPDATE {self.sfm_initial_pair_TN} " \
                                        f"SET " \
                                        f"first_image='{image_pair[0]}', " \
                                        f"second_image='{image_pair[1]}' " \
                                        f"WHERE session_id='{in_session_id}'"
            self.__sql_command.execute(update_initial_image_pair_str)
        else:
            insert_initial_image_pair_str = f"INSERT INTO {self.sfm_initial_pair_TN} " \
                                            f"(session_id, first_image, second_image) VALUES " \
                                            f"('{in_session_id}', '{image_pair[0]}', '{image_pair[1]}')"
            self.__sql_command.execute(insert_initial_image_pair_str)

        self.__database_conn.commit()

    def get_sfm_initial_pair(self, in_session_id):
        if self.__sql_command is None:
            print("Database not initialized yet!")
            return None
        get_initial_pair_cmd = f"SELECT first_image, second_image FROM {self.sfm_initial_pair_TN} " \
                               f"WHERE ID='{in_session_id}'"

        self.__sql_command.execute(get_initial_pair_cmd)
        ret_pair = self.__sql_command.fetchone()
        return ret_pair

    def set_images_path(self,  in_image_path: str, in_session_id: str):
        set_image_path_cmd = f"INSERT OR REPLACE INTO {self.session_management_TN} (ID, images_location)" \
                             f"VALUES({in_session_id}, '{in_image_path}')"
        self.__sql_command.execute(set_image_path_cmd)
        self.__database_conn.commit()

    def get_images_path(self, in_session_id: str):
        if self.__database_conn is None:
            return ''
        get_image_path_str = f"SELECT images_location FROM session_management " \
                             f"WHERE session_management.ID = {in_session_id}"
        self.__sql_command.execute(get_image_path_str)
        image_path = self.__sql_command.fetchone()
        if len(image_path) > 0:
            return image_path[0]


    def new_session(self, user_name, user_location, images_location, client_db_location) -> str:
        session_id = '0'
        insert_session_str = f"INSERT INTO {self.session_management_TN} (user, user_location, images_location," \
                             f"client_db_location, status_new_session) " \
                             f"VALUES('{user_name}','{user_location}', '{images_location}', '{client_db_location}', 1)"
        self.__sql_command.execute(insert_session_str)
        try:
            self.__database_conn.commit()
            session_id = str(self.__sql_command.lastrowid)
            return session_id
        except:
            return 'Failure'

    def get_new_session_id(self):
        insert_session_str = f""
        return '22'

    def is_session_exists(self, in_session_id):
        check_session_id_str = f"SELECT EXISTS (SELECT ID FROM {self.session_management_TN} WHERE ID='{in_session_id}')"
        self.__sql_command.execute(check_session_id_str)
        exists = self.__sql_command.fetchone()
        if exists[0] == 1:
            return True
        else:
            return False

    def __connect_to_maya_database(self):
        maya_database_name = f"E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_" \
                             f"Update/temp_data_database/MayaOPDatabase.sqlite"
        self.__client_db_conn = sql3.connect(maya_database_name)
        self.__client_sql_command = self.__client_db_conn.cursor()


    def store_user_selected_ponits(self, in_session_id):
        # try:
        if self.__client_db_conn is None:
            self.__connect_to_maya_database()

        clear_user_selected_pts_str = f"DELETE FROM {self.__user_selected_point_TN} WHERE session_id={in_session_id}"
        self.__sql_command.execute(clear_user_selected_pts_str)
        self.__database_conn.commit()

        get_correspondIDs_str = "SELECT DISTINCT TagPointsInScene.CorrespondingID FROM TagPointsInScene"
        self.__client_sql_command.execute(get_correspondIDs_str)
        correspondIDs = self.__client_sql_command.fetchall()

        get_image_names_str = f"SELECT DISTINCT TagPointsInImages.ImageFileName FROM TagPointsInImages " \
                              f"ORDER BY TagPointsInImages.ImageFileName"
        self.__client_sql_command.execute(get_image_names_str)
        image_filenames = self.__client_sql_command.fetchall()
        image_one_filename = image_filenames[0][0]
        image_two_filename = image_filenames[1][0]

        user_selected_points_data = []
        for correspond_id in correspondIDs:
            cid = correspond_id[0]
            get_image_one_xy_str = f"SELECT TagPointsInImages.ImagePoint_x, TagPointsInImages.ImagePoint_y from " \
                                   f"TagPointsInImages where TagPointsInImages.ImageFileName = '{image_one_filename}' " \
                                   f"and TagPointsInImages.CorrespondingID = {cid}"
            get_image_two_xy_str = f"SELECT TagPointsInImages.ImagePoint_x, TagPointsInImages.ImagePoint_y from " \
                                   f"TagPointsInImages where TagPointsInImages.ImageFileName = '{image_two_filename}' " \
                                   f"and TagPointsInImages.CorrespondingID = {cid}"
            get_world_xyz_str = f"SELECT TagPointsInScene.ScenePoint_x, TagPointsInScene.ScenePoint_y, " \
                                f"TagPointsInScene.ScenePoint_z FROM TagPointsInScene " \
                                f"WHERE TagPointsInScene.CorrespondingID = {cid}"
            self.__client_sql_command.execute(get_image_one_xy_str)
            image_one_xy = self.__client_sql_command.fetchone()
            self.__client_sql_command.execute(get_image_two_xy_str)
            image_two_xy = self.__client_sql_command.fetchone()
            self.__client_sql_command.execute(get_world_xyz_str)
            world_xyz = self.__client_sql_command.fetchone()
            user_selected_points_data.append((in_session_id, cid, world_xyz[0], world_xyz[1], world_xyz[2],
                                              image_one_filename, image_one_xy[0], image_one_xy[1],
                                              image_two_filename, image_two_xy[0], image_two_xy[1]))

        insert_point_data_str = f"INSERT INTO {self.__user_selected_point_TN} " \
                                f"(session_id, point_id, world_x, world_y, world_z, image_one_name, image_one_x," \
                                f"image_one_y, image_two_name, image_two_x, image_two_y) " \
                                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self.__sql_command.executemany(insert_point_data_str, user_selected_points_data)
        self.__database_conn.commit()

        # except:
        #     print("Copy user selected points failure!")
        #     return False

        return True

    def set_image_pair(self, in_session_id:str, image_pair: tuple):
        set_image_pair_str = f"INSERT OR REPLACE INTO sfm_initial_pair (ID, session_id, first_image, second_image) " \
                             f"VALUES((SELECT ID FROM sfm_initial_pair WHERE session_id={in_session_id}), " \
                             f"{in_session_id}, {image_pair[0]}, {image_pair[1]});"
        self.__sql_command.execute(set_image_pair_str)
        self.__database_conn.commit()

    def get_sfm_result(self):
        ...

    def get_ground_true(self, in_select_image: int) -> GroundTrueList:
        if self.__database_conn is None:
            return None
        ground_true = GroundTrueList()
        if in_select_image == 1:
            select_image_column = 'image_one'
        elif in_select_image == 2:
            select_image_column = 'image_two'

        get_ground_true_str = f"SELECT point_id, world_x, world_y, world_z, {select_image_column}_x, " \
                              f"{select_image_column}_y FROM {self.__user_selected_point_TN}"

        for row in self.__sql_command.execute(get_ground_true_str):
            new_ground_true = GroundTrue()
            new_ground_true.point_id = row[0]
            new_ground_true.world_point_3d = np.array([row[1], row[2], row[3]])
            new_ground_true.image_point_2d = np.array([row[4], row[5]])
            ground_true.ground_true_list.append(new_ground_true)

        return ground_true

    def close_server_database(self):
        self.__database_conn.close()

    def close_client_database(self):
        self.__client_db_conn.close()

