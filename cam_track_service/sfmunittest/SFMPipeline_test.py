import os
import numpy as np
import matplotlib.pyplot as plt
from SFMOperations import SFMPipeline as SFMpl
from SFMOperations import TwoViewCalculator
from SFMOperations import N_ViewCalculator as NVC

image_file_dir = r"E:\VHQ\camera-track-study\track-with-model\CameraTrackModelAlignment_Py38_Algorithm_Update\UnitTest\openCVAPITest\test-image"
focalLength = 933.3352
SFM_pipeline = SFMpl.SFMPipeLine()
SFM_pipeline.set_images_dir(image_file_dir)
SFM_pipeline.define_camera_intrinsic(focalLength)
SFM_pipeline.construct_sfm_image_list()
SFM_pipeline.calculate_camera_Rt_to_pre()
SFM_pipeline.calculate_Rt_to_cam0()
SFM_pipeline.construct_image_trajectory_pairs(1)
SFM_pipeline.construct_trajectories()
SFM_pipeline.construct_triangulate_data()

SFM_pipeline.filter_feature_in_shadow()

sfm_image_list = SFM_pipeline.sfm_pipeline_data.sfm_images

for a_sfm_image in sfm_image_list:
    print("============================================")
    print(a_sfm_image.dump_matched_indices_to_pre())
    print(a_sfm_image.dump_matched_indices_to_next())
