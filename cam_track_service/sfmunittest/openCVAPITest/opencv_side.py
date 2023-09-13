import cv2
import numpy as np
import os
K = np.array([[933.3352, 0., 480.],
            [0., 933.3352, 270.],
            [0., 0., 1.]])

imageFileForlder = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment/UnitTest/openCVAPITest/test-image/"

fileNameList = os.listdir(imageFileForlder)
fileList = [imageFileForlder + aFileName for aFileName in fileNameList]

print(sfm.reconstruct(images = fileList, K = K, is_projective = True))
