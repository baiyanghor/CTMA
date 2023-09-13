import os
import numpy as np
from SFMOperations import ManualDataMixin
from Configure import ConfigureOperators


class MayaSFMData(object):
    def __init__(self, image_file_list):
        self.image_ordered_file_list = image_file_list
        self.manual_data = None

    def construct_manual_data(self):
        workdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        conf_ops = ConfigureOperators.ConfigureOps(workdir)
        database_file = os.path.join(workdir, conf_ops.getMayaInterfaceDatabaseName())
        data_query_str = "SELECT * from TagPointsInImages ORDER by FrameNumber, CorrespondingID"
        manual_data_handler = ManualDataMixin.ManualData(database_file, data_query_str)
        manual_data_handler.construct_from_db()
        self.manual_data = manual_data_handler.get_image_data_list()

    def get_ordered_image_indices(self):
        image_file_list = []

        for a_image_data in self.manual_data:
            image_file_list.append(a_image_data.image_file_name)

        image_file_indices = []
        for index, a_image_file_name in enumerate(self.image_ordered_file_list):
            if os.path.basename(a_image_file_name) in image_file_list:
                image_file_indices.append(index)

        return image_file_indices

    def get_corresponding_points(self, image_index: int):
        points = []
        corresponding_point_data = self.manual_data[image_index].get_corresponding_points()

        for a_point in corresponding_point_data:
            points.append([a_point['x'], a_point['y']])

        return np.asarray(points)
