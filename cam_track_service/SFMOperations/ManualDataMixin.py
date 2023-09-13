from typing import List
import sqlite3
"""
corresponding_point:
{
    corresponding_id: int
    x: int
    y: int
}
"""


class ImageData(object):
    def __init__(self, image_file_name):
        self.image_file_name = image_file_name
        self.corresponding_points = []

    def add_corresponding_point(self, corresponding_id, x, y):
        self.corresponding_points.append({'corresponding_id': corresponding_id, 'x': x, 'y': y})

    def get_corresponding_points(self):
        return self.corresponding_points


class ManualData(object):
    def __init__(self, database_file, data_query_str):
        self.image_data_list = []
        self.database_file = database_file
        self.data_query_str = data_query_str

    def set_database_file(self, database_file, data_query_str):
        self.database_file = database_file
        self.data_query_str = data_query_str

    def add_image(self, image_data: ImageData):
        self.image_data_list.append(image_data)

    def get_image_data_list(self) -> List[ImageData]:
        return self.image_data_list

    def construct_from_db(self):
        db_conn = sqlite3.connect(self.database_file)
        cur = db_conn.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute(self.data_query_str)
        dataset = cur.fetchall()
        if len(dataset) > 0:
            self.image_data_list = []
            last_image_file_name = dataset[0]['ImageFileName']
            a_image = ImageData(last_image_file_name)
            self.add_image(a_image)

            for row in dataset:
                last_image_file_name = row['ImageFileName']
                if a_image.image_file_name != last_image_file_name:
                    a_image = ImageData(row['ImageFileName'])
                    self.add_image(a_image)
                a_image.add_corresponding_point(row['CorrespondingID'], row['ImagePoint_x'], row['ImagePoint_y'])

        db_conn.close()

    def dump(self):
        for a_image in self.image_data_list:
            print(a_image.image_file_name)
            print(a_image.corresponding_points)


