import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.SFMCalculationNodes import MaskingFeatures
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/cam_track_service"
session_id = "61aed9e316545f550540ddc3"

class SelectThis:
    one = 1
    two = 2


# mask_ops = MaskingFeatures(session_id=session_id, work_dir=work_dir)

# mask_ops.generate_mask()
# mask_ops.masking_features()
#

db_ops = MongodbInterface(work_dir)
sfm_image_list = db_ops.get_sfm_image_list(session_id)

mat_one = cv.imread(sfm_image_list[SelectThis.one].image_file_name)
mat_two = cv.imread(sfm_image_list[SelectThis.two].image_file_name)

one_match_to_next = sfm_image_list[SelectThis.one].dump_matches_to_next()
two_match_to_pre = sfm_image_list[SelectThis.two].dump_matches_to_pre()

colorsForLine = []
for p1, p2 in zip(one_match_to_next, two_match_to_pre):
    color = np.random.randint(0, 255, 3).tolist()
    resImage1 = cv.circle(mat_one, p1.astype('int').tolist(), 1, color, 1)
    resImage2 = cv.circle(mat_two, p2.astype('int').tolist(), 1, color, 1)
    colorsForLine.append(color)

# res = np.concatenate((mat_one, mat_two), axis=1)
# for i, p in enumerate(zip(one_match_to_next, two_match_to_pre)):
#     res = cv.line(res, p[0].astype('int').tolist(), [int(p[1][0]) + mat_one.shape[1], int(p[1][1])], colorsForLine[i], 2)

res = np.concatenate((mat_one, mat_two), axis=0)
# for i, p in enumerate(zip(one_match_to_next, two_match_to_pre)):
#     res = cv.line(res, p[0].astype('int').tolist(), [int(p[1][0]), int(p[1][1]) + mat_one.shape[0]], colorsForLine[i], 2)

# cv.imshow('Result', res)
# cv.waitKey()

show_result = cv.cvtColor(res, cv.COLOR_BGR2RGB)
plt.imshow(show_result)
plt.show()

