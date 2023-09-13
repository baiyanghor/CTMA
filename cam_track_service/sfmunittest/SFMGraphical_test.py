from SFMOperations import SFMPipeline as SFMpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
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
# true_intrinsic = SFM_pipeline.sequence_intrinsic.copy()
# true_intrinsic[0,2] = SFM_pipeline.sequence_image_size[1]/2.0
# true_intrinsic[1,2] = SFM_pipeline.sequence_image_size[0]/2.0

def get_camera_positions():
    camera_centers_list = []
    ids = []
    camera_rotation_list = []
    for a_cam in SFM_pipeline.sfm_pipeline_data.sfm_data.cameras:
        camera_centers_list.append(a_cam.euc_to_cam_0.flatten())
        ids.append(str(a_cam.camera_id))
        # camera_rotation_list.append(cv2.Rodrigues(a_cam.R_to_cam_0)[0].flatten())
        camera_rotation_list.append(a_cam.R_to_cam_0[:,2])
    the_camera_centers = np.asarray(camera_centers_list)
    the_camera_rotations = np.asarray(camera_rotation_list)
    return ids, the_camera_centers, the_camera_rotations

def get_world_points():
    world_point_3d = []
    for a_trajectory in SFM_pipeline.sfm_pipeline_data.sfm_data.trajectories:
        world_point_3d.append(a_trajectory.world_point.position_3d)
    return np.array(world_point_3d)


c_ids, camera_centers, camera_rotations = get_camera_positions()
world_points_3d = get_world_points()
w_x = world_points_3d[:, 0]
w_y = world_points_3d[:, 1]
w_z = world_points_3d[:, 2]

c_x = camera_centers[:, 0]
c_y = camera_centers[:, 1]
c_z = camera_centers[:, 2]
rv_x = camera_rotations[:,0]
rv_y = camera_rotations[:,1]
rv_z = camera_rotations[:,2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i, center in zip(c_ids, camera_centers):
    ax.text(center[0], center[1], center[2], str(i), None)

ax.quiver3D(c_x, c_y, c_z, rv_x, rv_y, rv_z, length=1, normalize=True)
ax.plot3D(c_x, c_y, c_z, color='gray')
# ax.scatter(c_x, c_y, c_z, marker='o')
# ax.scatter(w_x, w_y, w_z, marker='o')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_xlim3d([-20, 20])
# ax.set_ylim3d([-20, 20])
# ax.set_zlim3d([-20, 20])
# ax.set_xlim3d([-5, 5])
# ax.set_ylim3d([-5, 5])
# ax.set_zlim3d([-5, 5])
# ax.set_xlim3d([-1, 12])
# ax.set_ylim3d([-1, 12])
# ax.set_zlim3d([-1, 12])
# ax.view_init(-45,270)
# for angle in range(0, 360):
#     ax.view_init(angle, -30)
#     plt.draw()
#     plt.pause(.001)
plt.show()






