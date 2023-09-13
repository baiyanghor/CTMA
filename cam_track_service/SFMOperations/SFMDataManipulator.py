import numpy as np
from copy import deepcopy
from SFMOperations.MiscParser import parseStruct, getIndexOfRow, fullTest, checkIfPerm
from SFMOperations.vgg_Functions import vgg_X_from_xP_nonlin
from SFMOperations.BundleAjust import bundleAdjustment
import matplotlib.pyplot as plt
from FileOperations import TextFileOps



class SFMDataIterator(object):
    def __init__(self):
        self.frames = []
        self.focalLength = None
        self.Camera_Rts = None
        self.worldPoints = None
        self.imagePoints = None
        self.imagePointIDs = None
        self.intrinsicMatrix = np.array([1])
        self.denseMatch = None
        self.sparseMatches = None

    def closeEnought(self, other, tol):
        t1 = (self.frames == other.frames)
        t2 = (other.focalLength == self.focalLength)
        t3 = fullTest(self.Camera_Rts, other.Camera_Rts, tol)
        t4 = fullTest(self.worldPoints, other.worldPoints, tol)
        t5 = checkIfPerm(self.imagePoints, other.imagePoints)

        AllValuesFromIndexsA = self.imagePoints[self.imagePointIDs.astype(np.int), :]
        AllValuesFromIndexsB = other.imagePoints[other.imagePointIDs.astype(np.int), :]
        t6 = True
        for matchA in AllValuesFromIndexsA:
            fail = True
            for matchB in AllValuesFromIndexsB:
                if checkIfPerm(matchA, matchB):
                    fail = False
                    break
            if fail:
                t6 = False
        t7 = fullTest(self.intrinsicMatrix, other.intrinsicMatrix)

        return t1 and t2 and t3 and t4 and t5 and t6 and t7

    def __eq__(self, other):
        return self.closeEnought(other, 1e-02)


def createSFMIterator(frameID_A, frameID_B, intrinsicMatrix, pts_A, pts_B, Rt_A_B, focalLength):
    newSFMIterator = SFMDataIterator()
    newSFMIterator.frames = [frameID_A, frameID_B]
    newSFMIterator.intrinsicMatrix = intrinsicMatrix
    newSFMIterator.focalLength = focalLength
    newSFMIterator.Camera_Rts = np.zeros((3, 4, 2))
    pointsCount = pts_A.shape[0]

    newSFMIterator.Camera_Rts[:, :, 0] = np.hstack([np.eye(3), np.zeros((3, 1))])
    newSFMIterator.Camera_Rts[:, :, 1] = Rt_A_B

    newSFMIterator.worldPoints = np.zeros((pointsCount, 3))
    newSFMIterator.sparseMatches = np.hstack([pts_A, pts_B])
    newSFMIterator.imagePoints = np.vstack([pts_A, pts_B])

    newSFMIterator.imagePointIDs = np.zeros((pointsCount, 2), dtype=np.int)
    newSFMIterator.imagePointIDs[:, 0] = range(pointsCount)
    newSFMIterator.imagePointIDs[:, 1] = range(pointsCount, 2 * pointsCount)

    return newSFMIterator

def triangulate_two_view():
    pass

def triangulateReconstruction(aSFMDataIterator, imageSize):
    tempIterator = aSFMDataIterator
    worldPointsCount = tempIterator.worldPoints.shape[0]
    X = np.zeros((worldPointsCount, 4))
    colIms = np.array([imageSize[1], imageSize[0]]).reshape((2, 1))
    imageSizeList = colIms.repeat(len(aSFMDataIterator.frames), axis=1)

    # Iterate over world points
    for i in range(worldPointsCount):
        validCameraIDs = np.where(aSFMDataIterator.imagePointIDs[i] != -1)[0]
        P = np.zeros((3, 4, validCameraIDs.shape[0]))
        x = np.zeros((validCameraIDs.shape[0], 2))

        # Get the points on the camera plane and the projection matrix
        for cnt, aCameraID in enumerate(validCameraIDs):
            x[cnt, :] = tempIterator.imagePoints[tempIterator.imagePointIDs[i][aCameraID], :]
            P[:, :, cnt] = np.dot(tempIterator.intrinsicMatrix, tempIterator.Camera_Rts[:, :, aCameraID])

        X[i, :] = vgg_X_from_xP_nonlin(x, P, imageSizeList, X=None)

    allscales = X[:, 3].reshape((worldPointsCount, 1))
    tempIterator.worldPoints = X[:, 0:3] / np.hstack([allscales, allscales, allscales])

    return tempIterator


def mergeSFMIterators(SFMIteratorList, imageSize):
    mergedIterator = SFMIteratorList[0]

    # partial view merge
    for i in range(len(SFMIteratorList) - 1):
        print("Debug: Merging frame:")
        print(mergedIterator.frames)
        print(SFMIteratorList[i + 1].frames)
        mergedIterator = optimizedMergeSFMIteratorPair(mergedIterator, SFMIteratorList[i + 1], imageSize)
        print("Debug: Merging {0:d}th mergeSFMIterator".format(i))
        print("--------------------------------")

    return mergedIterator


def optimizedMergeSFMIteratorPair(SFMIterator_A, SFMIterator_B, imsize):
    mergedIterator = mergeSFMIteratorPair(SFMIterator_A, SFMIterator_B)
    mergedIterator = triangulateReconstruction(mergedIterator, imsize)
    mergedIterator = bundleAdjustment(mergedIterator, False)
    mergedIterator = removeOutlierPts(mergedIterator, 10)
    mergedIterator = bundleAdjustment(mergedIterator, False)
    return mergedIterator


def mergeSFMIteratorPair(SFMIterator_A, SFMIterator_B):
    # As an example are frames A 1 2 and B 2 3
    # the frames are the cameras or photos

    # Frames that overlap are first calculated
    overlappedFrames = list(set(SFMIterator_A.frames).intersection(SFMIterator_B.frames))

    # Then those that are proper to A and those proper to B (in eg A1 B3)
    nonoverlappingFrames_B = list(set(SFMIterator_B.frames).difference(SFMIterator_A.frames))
    nonoverlappingFIDs_B = [SFMIterator_B.frames.index(FNum) for FNum in nonoverlappingFrames_B]

    # If there are no commons, it returns error
    # If B's own are no mistake strip
    if len(overlappedFrames) == 0:
        # raise Exception("Comunes vacio ")
        raise Exception("Commons empty ")

    if len(nonoverlappingFrames_B) == 0:
        # raise Exception("No hay propias de B")
        raise Exception("There is no own of B")

    # Create mixed aSFMDataIterator equal to aSFMDataIterator A
    mergedIterator = deepcopy(SFMIterator_A)

    # For the first overlap (there may be many)
    firstOverlappedFrame = overlappedFrames[0]

    # Transform B.Camera_Rts b.worldPoints to the same coordinate system of A
    firstOverlappedFID_A = SFMIterator_A.frames.index(firstOverlappedFrame)
    firstOverlappedFID_B = SFMIterator_B.frames.index(firstOverlappedFrame)

    # Get rtB transformed
    cameraFrameTrans_B_A = concatenate_Rts(inverse_Rt(SFMIterator_A.Camera_Rts[:, :, firstOverlappedFID_A]),
                                           SFMIterator_B.Camera_Rts[:, :, firstOverlappedFID_B])
    # Apply a worldPoints B
    SFMIterator_B.worldPoints = transformWorldPointsby_Rt(np.transpose(SFMIterator_B.worldPoints),
                                                          cameraFrameTrans_B_A, False)
    # Mot is now the conchain of Camera_Rts and the inverse RtB
    for i in range(len(SFMIterator_B.frames)):
        SFMIterator_B.Camera_Rts[:, :, i] = concatenate_Rts(SFMIterator_B.Camera_Rts[:, :, i],
                                                            inverse_Rt(cameraFrameTrans_B_A))

    mergedIterator.frames = list(set(SFMIterator_A.frames).union(set(SFMIterator_B.frames)))

    tempCamera_Rts = np.zeros((3, 4, len(mergedIterator.frames)))
    tempCamera_Rts[:, :, np.array(range(len(SFMIterator_A.frames)))] = SFMIterator_A.Camera_Rts
    tempCamera_Rts[:, :, np.array(range(len(SFMIterator_A.frames), len(mergedIterator.frames)))] \
        = SFMIterator_B.Camera_Rts[:, :, nonoverlappingFIDs_B]

    mergedIterator.Camera_Rts = tempCamera_Rts
    # Add frames to chart
    # Now common frames case more than one

    for aFrame in overlappedFrames:
        frameID_A = SFMIterator_A.frames.index(aFrame)
        frameID_B = SFMIterator_B.frames.index(aFrame)

        imagePointIDs_A = SFMIterator_A.imagePointIDs[:, frameID_A]
        imagePoints_A = SFMIterator_A.imagePoints[imagePointIDs_A, :]

        imagePointIDs_B = SFMIterator_B.imagePointIDs[:, frameID_B]
        imagePoints_B = SFMIterator_B.imagePoints[imagePointIDs_B, :]

        samePointIDs_A = getSamePointsID_in_First(imagePoints_A, imagePoints_B)[0]
        samePoints_A = imagePoints_A[samePointIDs_A]

        samePointIDs_B = np.array([getIndexOfRow(imagePoints_B, row)[0][0] for row in samePoints_A])

        samePointIDs_A, samePointIDs_B = \
            deleteRepeatedPointIDs(samePointIDs_A.tolist(), samePointIDs_B.tolist(), imagePoints_A, imagePoints_B)

        samePointIDs_A = np.array(samePointIDs_A)
        samePointIDs_B = np.array(samePointIDs_B)
        # Check position in image?
        # Same points in images
        # Iterate on each point IDs
        for i in range(samePointIDs_A.shape[0]):
            # j for frame ID of none overlapping in iterator B
            for j in range(len(nonoverlappingFIDs_B)):
                sameFramePointID_B = SFMIterator_B.imagePointIDs[samePointIDs_B[i], nonoverlappingFIDs_B[j]]
                # I add an element to imagePoints and imagePointIDs
                mergedIterator.imagePoints = np.vstack([mergedIterator.imagePoints,
                                                        SFMIterator_B.imagePoints[sameFramePointID_B, :]])
                # Full the rest with negative ones?
                while mergedIterator.imagePointIDs.shape[1] < (len(SFMIterator_A.frames) + j + 1):
                    mergedIterator.imagePointIDs = np.hstack([mergedIterator.imagePointIDs,
                                                              minusOneMatrix(
                                                                  (mergedIterator.imagePointIDs.shape[0], 1))])
                # Confusing ?
                mergedIterator.imagePointIDs[samePointIDs_A[i], len(SFMIterator_A.frames) + j] = \
                    mergedIterator.imagePoints.shape[0] - 1

        # Calculate set difference
        differentPoints_B = getDifferentPointsInFirst(imagePoints_B, imagePoints_A)
        differentPointIDs_B = np.array([getIndexOfRow(imagePoints_B, row)[0][0] for row in differentPoints_B])
        # Iterate on point IDs
        for i in range(differentPointIDs_B.shape[0]):
            diffPointIDs_InFrame_B = SFMIterator_B.imagePointIDs[differentPointIDs_B[i], frameID_B]
            mergedIterator.imagePoints = np.vstack([mergedIterator.imagePoints,
                                                    SFMIterator_B.imagePoints[diffPointIDs_InFrame_B, :]])
            mergedIterator.imagePointIDs = np.vstack([mergedIterator.imagePointIDs,
                                                      minusOneMatrix((1, mergedIterator.imagePointIDs.shape[1]))])
            mergedIterator.imagePointIDs[mergedIterator.imagePointIDs.shape[0] - 1, frameID_A] = \
                mergedIterator.imagePoints.shape[0] - 1
            mergedIterator.worldPoints = np.vstack([mergedIterator.worldPoints,
                                                    SFMIterator_B.worldPoints[:, differentPointIDs_B[i]].reshape(
                                                        (1, 3))])

            for j in range(len(nonoverlappingFIDs_B)):
                diffPointIDs_InDiffFrame_B = SFMIterator_B.imagePointIDs[differentPointIDs_B[i],
                                                                         nonoverlappingFIDs_B[j]]
                mergedIterator.imagePoints = np.vstack([mergedIterator.imagePoints,
                                                        SFMIterator_B.imagePoints[diffPointIDs_InDiffFrame_B, :]])

                while mergedIterator.imagePointIDs.shape[1] < (len(SFMIterator_A.frames) + j + 1):
                    mergedIterator.imagePointIDs = np.hstack([mergedIterator.imagePointIDs,
                                                              minusOneMatrix(
                                                                  (mergedIterator.imagePointIDs.shape[0], 1))])

                mergedIterator.imagePointIDs[-1, len(SFMIterator_A.frames) + j] = \
                    mergedIterator.imagePoints.shape[0] - 1

    # Check if any column is left without any value
    # Select in imagePointIDs the common frames and diff in Gb
    # Make sure that for no point A and B is met
    # Being A = In common columns all have the value of -1
    # Where B = In diff columns the sum of the values greater than -1 is greater than 0

    newB = np.zeros((1, len(SFMIterator_B.frames)), dtype=int)
    newB[:, np.array(nonoverlappingFIDs_B)] = 1
    A = np.sum(SFMIterator_B.imagePointIDs[:, np.bitwise_not(newB.astype(np.bool)).astype(int)[0]], axis=1) < 0
    B = np.sum(SFMIterator_B.imagePointIDs[:, newB[0].astype(np.int)], axis=1) > 0
    assert (not np.any(np.bitwise_and(A, B)))

    return mergedIterator


def deleteRepeatedPointIDs(indices_A, indices_B, values_A, values_B):
    toDelete = []
    for i in range(len(indices_A)):
        for j in range(i + 1, len(indices_A)):
            if np.array_equal(values_A[indices_A[i]], values_A[indices_A[j]]):
                toDelete.append(indices_A[j])

    toDelete = list(set(toDelete))
    # Erase repeated but throw error.

    for e in toDelete:
        ind = indices_A.index(e)
        indices_A.remove(e)
        del indices_B[ind]

    assert (np.array_equal(values_A[np.array(indices_A).astype(int)],
                           values_B[np.array(indices_B).astype(int)]))
    assert (len(set(indices_A)) == len(indices_A))
    assert (len(set(indices_B)) == len(indices_B))
    assert (len(indices_B) == len(indices_A))

    return indices_A, indices_B


def uniqueRows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]


def uniqueRowsIndexs(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return idx


def minusOneMatrix(shape):
    return -1 * np.ones(shape, dtype=np.int32)


def getDifferentPointsInFirst(findInPoints, withoutPoints):
    findIn_rows = findInPoints.view([('', findInPoints.dtype)] * findInPoints.shape[1])
    without_rows = withoutPoints.view([('', withoutPoints.dtype)] * withoutPoints.shape[1])
    return np.setdiff1d(findIn_rows, without_rows).view(findInPoints.dtype).reshape(-1, findInPoints.shape[1])


def getSamePointsID_in_First(findInPoints, withinPoints):
    findInPoints = findInPoints.astype(np.float64)
    withinPoints = withinPoints.astype(np.float64)
    return np.nonzero(np.isin(findInPoints.view('d,d').reshape(-1), withinPoints.view('d,d').reshape(-1)))


def concatenate_Rts(Rt_1, Rt_2):
    R_1 = Rt_1[:, 0:3]
    t_2 = Rt_1[:, 3].reshape(3, 1)
    return np.hstack([np.dot(R_1, Rt_2[:, 0:3]), np.dot(R_1, Rt_2[:, 3]).reshape(3, 1) + t_2])


def inverse_Rt(Rt):
    inversed_R = np.transpose(Rt[0:3, 0:3])
    return np.hstack([inversed_R, np.dot(-inversed_R, Rt[0:3, 3]).reshape(3, 1)])


def removeOutlierPts(inSFMIterator, th_pix=10):
    sq_th_pix = th_pix * th_pix
    td = 2
    tincos = np.cos(np.pi * td * 1.0 / 180)
    for i in range(inSFMIterator.imagePointIDs.shape[1]):
        X = np.dot(inSFMIterator.intrinsicMatrix,
                   transformWorldPointsby_Rt(np.transpose(inSFMIterator.worldPoints),
                                             inSFMIterator.Camera_Rts[:, :, i], False))
        xy = X[0:2, :] / X[[2, 2], :]
        selector = np.where(inSFMIterator.imagePointIDs[:, i] != -1)[0]

        dif = xy[:, selector] - np.transpose(
            inSFMIterator.imagePoints[inSFMIterator.imagePointIDs[selector, i]])
        outliers = np.sum(dif * dif, axis=0) > sq_th_pix
        cantB = np.sum(outliers)
        if cantB > 0:
            print("I erased {0:f} outliers of {1:f} total points with sq_th_pix of {2:+f}".format(cantB,
                                                                                                     outliers.shape[0],
                                                                                                     sq_th_pix))
        p2keep = np.ones((1, inSFMIterator.worldPoints.shape[0]))
        p2keep[:, selector[outliers]] = False
        p2keep = p2keep[0].astype(np.bool)
        inSFMIterator.worldPoints = inSFMIterator.worldPoints[p2keep, :]
        inSFMIterator.imagePointIDs = inSFMIterator.imagePointIDs[p2keep, :]

    nF = len(inSFMIterator.frames)
    pos = np.zeros((3, nF))

    for ii in range(nF):
        Rt = inSFMIterator.Camera_Rts[:, :, ii]
        pos[:, ii] = -np.dot(np.transpose(Rt[0:3, 0:3]), Rt[:, 3])

    view_dirs = np.zeros((inSFMIterator.worldPoints.shape[0], 3, nF))

    for c in range(inSFMIterator.imagePointIDs.shape[1]):
        selector = np.where(inSFMIterator.imagePointIDs[:, c] != -1)[0]
        t = np.repeat(pos[:, c].reshape((1, 3)), inSFMIterator.worldPoints[selector, :].shape[0], axis=0)
        camera_v_d = inSFMIterator.worldPoints[selector, :] - t
        d_length = np.sqrt(np.sum(camera_v_d * camera_v_d, axis=1))
        dt = 1.0 / d_length
        camera_v_d = camera_v_d * np.transpose(np.vstack([dt, dt, dt]))
        view_dirs[selector, :, c] = camera_v_d

    for c1 in range(inSFMIterator.imagePointIDs.shape[1]):
        for c2 in range(inSFMIterator.imagePointIDs.shape[1]):
            if c1 == c2:
                continue
            selector = np.where(np.bitwise_and(inSFMIterator.imagePointIDs[:, c1] != -1,
                                               inSFMIterator.imagePointIDs[:, c2] != -1))[0]
            v_d1 = view_dirs[selector, :, c1]
            v_d2 = view_dirs[selector, :, c2]
            cos_a = np.sum(v_d1 * v_d2, axis=1)
            outliers = cos_a > tincos

            cantB = np.sum(outliers)
            if cantB > 0:
                print("I erased {0:d} outliers of {1:d} total points with cost_thr of {2:f}".format(cantB,
                                                                                                       outliers.shape[
                                                                                                           0], tincos))
            p2keep = np.ones((1, inSFMIterator.worldPoints.shape[0]))
            p2keep[:, selector[outliers]] = False
            p2keep = p2keep[0].astype(np.bool)
            inSFMIterator.worldPoints = inSFMIterator.worldPoints[p2keep, :]
            inSFMIterator.imagePointIDs = inSFMIterator.imagePointIDs[p2keep, :]

    return inSFMIterator


def saveCameraRtWorldPts(aSFMIterator, cameraRtFileName, worldPointsFilename):
    txtOps = TextFileOps.TextFileOps()
    txtOps.saveFlattenData(aSFMIterator.Camera_Rts, cameraRtFileName)
    txtOps.saveFlattenData(aSFMIterator.worldPoints, worldPointsFilename)


def transformWorldPointsby_Rt(X3D, Rt, isInverse=True):
    displacement_t = np.repeat(Rt[:, 3, np.newaxis], X3D.shape[1], axis=1)

    if isInverse:
        Y3D = np.dot(np.transpose(Rt[:, 0:3]), (X3D - displacement_t))
    else:
        Y3D = np.dot(Rt[:, 0:3], X3D) + displacement_t
    return Y3D


def camera_shape(Rt, w, h, focalLength, scale):
    V = np.array([
        [0, 0,           0,           focalLength, -(w * 0.5),  (w * 0.5),   (w * 0.5),  -(w * 0.5)],
        [0, 0,           focalLength, 0,           -(h * 0.5),  -(h * 0.5),  (h * 0.5),   (h * 0.5)],
        [0, focalLength, 0,           0,           focalLength, focalLength, focalLength, focalLength]
    ])
    V = scale * V
    V = transformWorldPointsby_Rt(V, Rt, True)
    return V


def showGraph(aSFMDataIterator, imsize, getAxis=False):
    fig = plt.figure("Triangulate View")
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # draw cameras
    for i in range(aSFMDataIterator.Camera_Rts.shape[2]):
        V = camera_shape(aSFMDataIterator.Camera_Rts[:, :, i], imsize[1], imsize[0], aSFMDataIterator.focalLength, 0.001)
        xi, yi, zi = V[0, [0, 4]], V[1, [0, 4]], V[2, [0, 4]]
        ax.plot(xi, yi, zi)
        xi, yi, zi = V[0, [0, 5]], V[1, [0, 5]], V[2, [0, 5]]
        ax.plot(xi, yi, zi)
        xi, yi, zi = V[0, [0, 6]], V[1, [0, 6]], V[2, [0, 6]]
        ax.plot(xi, yi, zi)
        xi, yi, zi = V[0, [0, 7]], V[1, [0, 7]], V[2, [0, 7]]
        ax.plot(xi, yi, zi)
        ax.plot(V[0, [4, 5, 6, 7, 4]], V[1, [4, 5, 6, 7, 4]], V[2, [4, 5, 6, 7, 4]])

    ax.scatter(aSFMDataIterator.worldPoints[:, 0], aSFMDataIterator.worldPoints[:, 1],
               aSFMDataIterator.worldPoints[:, 2])

    if getAxis:
        return ax
    else:
        plt.show()
