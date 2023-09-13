import os
import numpy as np
from ImageOperations import OpenCVDataFromFileList
from FileOperations import ImageFileOperators
from FileOperations import TextFileOps
from SFMOperations import SFMCalculators
from SFMOperations import SFMDataManipulator
from SFMOperations.BundleAjust import bundleAdjustment
from Configure import ConfigureOperators
from SFMOperations import ConstructMayaSFMIterator
txtFileOps = TextFileOps.TextFileOps()
workdir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
confOps = ConfigureOperators.ConfigureOps(workdir)

fileOps = ImageFileOperators.ImageFileOperators()
imgDataOps = OpenCVDataFromFileList.OpenCVData()
sfmOps = SFMCalculators.SFMOperators()

# testImagesPath = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38/UnitTest/openCVAPITest/test-image"
testImagesPath = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38/UnitTest/TestImages"
# focalLength = 933.3352
focalLength = 719.5459
sfmOps.setImageFilesPath(testImagesPath)
sfmOps.setFocalLength(focalLength)
imageFilesList = fileOps.getImageRawFileList(testImagesPath)

imgDataOps.loadImageDataList(imageFilesList)

viewCount = len(imageFilesList)

# imgDataOps.initialFeatureExtractor('SIFT')
# imageCount = imgDataOps.saveCorrespondencesToFileList()

SFMIteratorList = []
sfmOps.setImageSize(imgDataOps.getImageSize())
imageSize = sfmOps.getImageSize()
intrinsicMatrix = sfmOps.initialCameraIntrinsic()

ADD_MANUAL_DATA = False

if ADD_MANUAL_DATA:
    manual_data_handler = ConstructMayaSFMIterator.MayaSFMData(imageFilesList)
    manual_data_handler.construct_manual_data()
    image_indices_for_additional_data = manual_data_handler.get_ordered_image_indices()

for i in range(viewCount - 1):
    pointsA, pointsB = imgDataOps.readCorrespondencesFromFile(i)

    if ADD_MANUAL_DATA:
        if i in image_indices_for_additional_data and i+1 in image_indices_for_additional_data:
            print(f"Debug: Add manual data in {i} and {i+1}")
            pointsA = np.vstack((pointsA, manual_data_handler.get_corresponding_points(i)))
            pointsB = np.vstack((pointsB, manual_data_handler.get_corresponding_points(i+1)))

    sfmOps.setCorrespondences(pointsA, pointsB)

    rtCameraPose = sfmOps.getCameraPose()

    aSFMIterator = SFMDataManipulator.createSFMIterator(i, i+1, intrinsicMatrix, pointsA, pointsB, rtCameraPose, focalLength)

    aSFMIterator = SFMDataManipulator.triangulateReconstruction(aSFMIterator, imageSize)

    aSFMIterator = bundleAdjustment(aSFMIterator, adjustFocalLength=False)

    SFMIteratorList.append(aSFMIterator)

mergedSFMIterator = SFMDataManipulator.mergeSFMIterators(SFMIteratorList, imageSize)

SFMDataManipulator.showGraph(mergedSFMIterator, imageSize)
#
# tempDataPath = confOps.getTempDataPath()
# cameraRtFileName = str(os.path.abspath(os.path.join(workdir, tempDataPath, "SFM_cameraRt.txt")))
# worldPtsFileName = str(os.path.abspath(os.path.join(workdir, tempDataPath, "SFM_worldPoints.txt")))
# intrinsicFileName = str(os.path.abspath(os.path.join(workdir, tempDataPath, "SFM_intrinsic.txt")))
#
# txtFileOps.saveFlattenData(sfmOps.getFullIntrinsicMatrix(), intrinsicFileName)
#
# SFMDataManipulator.saveCameraRtWorldPts(mergedSFMIterator, cameraRtFileName, worldPtsFileName)




