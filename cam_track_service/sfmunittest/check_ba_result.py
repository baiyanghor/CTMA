import matplotlib.pyplot as plt
import numpy as np

from SFMOperations.SFMDataTypes import BA_PHASE
from SFMOperations.SFMMongodbInterface import MongodbInterface
session_id = '61ca8f19e149e4556ff5f1f0'
work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_fordistrib/cam_track_service"
db_ops = MongodbInterface(work_dir)
ba_result = db_ops.get_ba_benchmark(session_id, BA_PHASE.BA_GLOBAL_ITERATIVE)
mean = np.empty(0)
std_variation = np.empty(0)
ba_round = np.empty(0)
for ba_idx, data in ba_result.items():
    ba_round = np.append(ba_round, int(ba_idx))
    mean = np.append(mean, data['mean'])
    std_variation = np.append(std_variation, data['stdvar'])

fig, axes = plt.subplots(2, 1)
fig.suptitle(f"Global Bundle Adjustment on Rounds")
axes[0].plot(ba_round, mean)
axes[0].grid(True)
axes[0].set_title("Mean Pixel Error")
axes[0].set_ylabel('Pixel')
axes[0].set_xlabel('SFM Image ID')

axes[1].plot(ba_round, std_variation)
axes[1].grid(True)
axes[1].set_title("Pixel Standard Variation")
axes[1].set_ylabel('Pixel')
axes[1].set_xlabel('SFM Image ID')
plt.show(); plt.ion()




