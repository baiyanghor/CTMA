import os
import numpy as np
from ImageOperations import OpenCVDataFromFileList
from FileOperations import ImageFileOperators
from FileOperations import TextFileOps
from SFMOperations import SFMCalculators
from SFMOperations import SFMDataManipulator
from SFMOperations.BundleAjust import bundleAdjustment
from Configure import ConfigureOperators

txtFileOps = TextFileOps.TextFileOps()
workdir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
confOps = ConfigureOperators.ConfigureOps(workdir)

fileOps = ImageFileOperators.ImageFileOperators()
imgDataOps = OpenCVDataFromFileList.OpenCVData()
sfmOps = SFMCalculators.SFMOperators()

testImagesPath = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38/UnitTest/openCVAPITest/test-image"
# testImagesPath = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38/UnitTest/TestImages"
focalLength = 933.335200004
# focalLength = 719.5459
sfmOps.setImageFilesPath(testImagesPath)
sfmOps.setFocalLength(focalLength)
imageFilesList = fileOps.getImageRawFileList(testImagesPath)
imgDataOps.loadImageDataList(imageFilesList)

viewCount = len(imageFilesList)

savePoints = 0

if savePoints:
    imgDataOps.initialFeatureExtractor('OF')
    imageCount = imgDataOps.saveCorrespondencesToFileList()

else:
    SFMIteratorList = []
    sfmOps.setImageSize(imgDataOps.getImageSize())
    imageSize = sfmOps.getImageSize()
    intrinsicMatrix = sfmOps.initialCameraIntrinsic()

    pointsA, pointsB = imgDataOps.readCorrespondencesFromFile(0)

    sfmOps.setCorrespondences(pointsA, pointsB)

    rtCameraPose = sfmOps.getCameraPose()

    aSFMIterator: SFMDataManipulator.SFMDataIterator = \
        SFMDataManipulator.createSFMIterator(0, 1, intrinsicMatrix, pointsA, pointsB, rtCameraPose, focalLength)
    print("Image Size")
    print(imageSize)
    aSFMIterator = SFMDataManipulator.triangulateReconstruction(aSFMIterator, imageSize)

    print("Debug Main: Before bundle adjustment")
    print(aSFMIterator.Camera_Rts[:, :, 0])
    print(aSFMIterator.Camera_Rts[:, :, 1])
    print("====================================")

    print(aSFMIterator.worldPoints)
    print("+++++++++++++++++++++++++++++++++++++")
    # aSFMIterator = bundleAdjustment(aSFMIterator, adjustFocalLength=False)
    SFMDataManipulator.showGraph(aSFMIterator, imageSize)



#
# SFMIteratorList = []
# sfmOps.setImageSize(imgDataOps.getImageSize())
# imageSize = sfmOps.getImageSize()
# intrinsicMatrix = sfmOps.initialCameraIntrinsic()
# for i in range(viewCount - 1):
#     print("Debug Main: Dealing with {:d}th Image".format(i + 1))
#     pointsA, pointsB = imgDataOps.readCorrespondencesFromFile(i)
#     print("Matched A {0:d} points, B {1:d} point".format(pointsA.shape[0], pointsB.shape[0]))
#     sfmOps.setCorrespondences(pointsA, pointsB)
#
#     rtCameraPose = sfmOps.getCameraPose()
#
#     aSFMIterator = SFMDataManipulator.createSFMIterator(i, i+1, intrinsicMatrix, pointsA, pointsB, rtCameraPose, focalLength)
#
#     aSFMIterator = SFMDataManipulator.triangulateReconstruction(aSFMIterator, imageSize)
#
#     aSFMIterator = bundleAdjustment(aSFMIterator, adjustFocalLength = False)
#
#     SFMIteratorList.append(aSFMIterator)
#
# mergedSFMIterator = SFMDataManipulator.mergeSFMIterators(SFMIteratorList, imageSize)
#
# SFMDataManipulator.showGraph(mergedSFMIterator, imageSize)