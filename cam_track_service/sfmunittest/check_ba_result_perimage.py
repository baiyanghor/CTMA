import matplotlib.pyplot as plt
import numpy as np
from SFMOperations.SFMMongodbInterface import MongodbInterface
from SFMOperations.SFMDataTypes import BA_PHASE

session_id = '61ca8f19e149e4556ff5f1f0'
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/cam_track_service"

db_ops = MongodbInterface(work_dir)
start_image_id = 3
end_image_id = 11
check_ba_round = 500

mean_list = np.empty(0)
stdvar_list = np.empty(0)
image_idx_list = list(range(start_image_id, end_image_id))

for image_idx in range(start_image_id, end_image_id):
    ba_data = db_ops.get_ba_residual_perimage(session_id, image_idx, BA_PHASE.BA_GLOBAL_ITERATIVE)
    mean_list = np.append(mean_list, ba_data[str(check_ba_round)]['mean'])
    stdvar_list = np.append(stdvar_list, ba_data[str(check_ba_round)]['stdvar'])

fig, axes = plt.subplots(2, 1)
fig.suptitle(f"Bundle adjustment round {check_ba_round}")
fig.tight_layout(pad=3.0)
axes[0].plot(image_idx_list, mean_list, 'bo-')
axes[0].grid(True)
axes[0].set_title("Mean Pixel Error")
axes[0].set_ylabel('-- Pixel --')
axes[0].set_xlabel('-- SFM Image ID --')

axes[1].plot(image_idx_list, stdvar_list, 'bo-')
axes[1].grid(True)
axes[1].set_title("Pixel Standard Variation")
axes[1].set_ylabel('-- Pixel --')
axes[1].set_xlabel('-- SFM Image ID --')
plt.show()


# print(mean_list.shape)
# print(stdvar_list.shape)
