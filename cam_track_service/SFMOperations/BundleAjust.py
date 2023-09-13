import numpy as np
from scipy.optimize import least_squares

def bundleAdjustment(inSFMIterator, adjustFocalLength = False):
    camCount = inSFMIterator.Camera_Rts.shape[2]
    tempCamera_Rts = np.zeros((3,2, camCount))

    for i in range(camCount):
        tempCamera_Rts[:,0,i] = rotationMatrix2angleaxis(inSFMIterator.Camera_Rts[:,0:3,i])
        tempCamera_Rts[:,1,i] = inSFMIterator.Camera_Rts[:,3,i]

    tempWorldPoints = inSFMIterator.worldPoints

    px, py = 0, 0
    currFocalLength = inSFMIterator.focalLength
    print("Debug bundleAdjustment:")
    print(inSFMIterator.imagePointIDs.shape)
    print("-------------------------------------")
    RR_Error = reprojectionResidual(inSFMIterator.imagePointIDs, inSFMIterator.imagePoints, px,
                                py, currFocalLength, tempCamera_Rts, tempWorldPoints)

    getErrorBenchmark = lambda x : 2 * np.sqrt(np.sum(np.power(x,2)) / x.shape[0])

    print("Initial error of", getErrorBenchmark(RR_Error))

    # Realize value optimization
    # I want to be able to minimize the norm of the reprojectionResidual vector    
    getResidual_Fix_F = lambda x : wrapResidul_Fix_F(x, camCount, inSFMIterator.imagePointIDs,
                                       inSFMIterator.imagePoints, px, py, currFocalLength)
    
    sol = least_squares(getResidual_Fix_F, packCamRtsWorldPts(tempCamera_Rts, tempWorldPoints), method='lm', max_nfev=1000)
    rtCameraRts, rtWorldPoints = unpackCamRtsWorldPts(sol.x, camCount, inSFMIterator.imagePointIDs.shape[0])

    print("Error after optimizing ", getErrorBenchmark(sol.fun))

    if adjustFocalLength:
        # Perform value optimization
        # I want to be able to minimize the norm of the reprojectionResidual vector
        getResidual = lambda x: wrapResidual(x, camCount, inSFMIterator.imagePointIDs, inSFMIterator.imagePoints, px, py)
        sol = least_squares(getResidual, packCamRtsWorldPtsFocal(tempCamera_Rts, tempWorldPoints, currFocalLength), method='lm')
        rtFocalLength, rtCameraRts, rtWorldPoints = unpackCamRtsWorldPtsFocal(sol.x, camCount, inSFMIterator.imagePointIDs.shape[0])
        print("Error after optimizing ".format(getErrorBenchmark(sol.fun)))
        inSFMIterator.intrinsicMatrix = np.eye(3) * rtFocalLength
        inSFMIterator.intrinsicMatrix[2, 2] = 1
        inSFMIterator.focalLength = rtFocalLength

    for i in range(camCount):
        inSFMIterator.Camera_Rts[:,:, i] = np.hstack([axisAngle2RotationMatrix(rtCameraRts[:, 0, i]),
                                                      rtCameraRts[:,1,i].reshape((3,1))])

    inSFMIterator.worldPoints = rtWorldPoints
    return inSFMIterator


def packCamRtsWorldPtsFocal(Camera_Rts,worldPts,focalLength):
    flattenCamRts = Camera_Rts.flatten(order='F')
    flattenWorldPts = worldPts.flatten(order='F')
    
    return np.concatenate((focalLength, flattenCamRts, flattenWorldPts))

def packCamRtsWorldPts(camera_Rts,worldPoints):    
    return np.concatenate((camera_Rts.flatten(order='F'), worldPoints.flatten(order='F')))

def wrapResidul_Fix_F(x, camCount, imagePointIDs, imagePoints, px, py, focalLength):
    camera_Rts, worldPoints = unpackCamRtsWorldPts(x, camCount, imagePointIDs.shape[0])
    
    return reprojectionResidual(imagePointIDs, imagePoints, px, py, focalLength, camera_Rts, worldPoints)

def wrapResidual(x,camCount,imagePointIDs, imagePoints, px, py):
    camera_Rts, worldPoints, focalLength = unpackCamRtsWorldPtsFocal(x, camCount, imagePointIDs.shape[0])
    
    return reprojectionResidual(imagePointIDs, imagePoints, px, py, focalLength, camera_Rts, worldPoints)

def rotationMatrix2angleaxis(R):
    #El problema ocurre que hay valores muy pequenos asi que se sigue el sigueinte proceso
    # The problem occurs that there are very small values so the following process is followed

    ax = [0,0,0]
    ax[0] = R[2,1] - R[1,2]
    ax[1] = R[0,2] - R[2,0]
    ax[2] = R[1,0] - R[0,1]
    ax = np.array(ax)


    costheta = max((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0, -1.0)
    costheta = min(costheta, 1.0)

    sintheta = min(np.linalg.norm(ax) * 0.5, 1.0)
    theta = np.arctan2(sintheta, costheta)
    # TODO (this had precision problems in matlab I don't know if they are maintained)
    # for security copy the version that fixes these problems

    kthreshold = 1e-12
    if (sintheta > kthreshold) or (sintheta < -kthreshold):
        r = theta / (2.0 *sintheta)
        ax = r * ax
        return ax
    else:
        if costheta > 0.0:
            ax = ax *0.5
            return ax
        inv_one_minus_costheta = 1.0 / (1.0 - costheta)

        for i in range(3):
            ax[i] = theta * np.sqrt((R[i, i] - costheta) * inv_one_minus_costheta)
            # ax[i] = theta * np.sqrt((R(i, i) - costheta) * inv_one_minus_costheta)
            # Changed by Bai Y. Hor
            cond1 = ((sintheta < 0.0) and (ax[i] > 0.0))
            cond2 = ((sintheta > 0.0) and (ax[i] < 0.0))
            if cond1 or cond2:
                ax[i] = -ax[i]
        return ax

# The reprojectionResidual function has 3 uses. 1 Evaluate a model, 2. Optimize given [Camera_Rts[:]; worldPoints[:]]
# And 3 optimize daod [focalLength; Camera_Rts (:); worldPoints (:)] That's why these unpack functions exist


def unpackCamRtsWorldPts(vect, camCount, imagePtCount):
    flattenCount = 3 * 2 * camCount
    camera_Rts = np.reshape(vect[0:flattenCount], (3, 2, camCount), order='F')
    worldPoints = np.reshape(vect[flattenCount:], (imagePtCount,3), order='F' )
    return camera_Rts, worldPoints


def unpackCamRtsWorldPtsFocal(vect, camCount, imagePtsCount):
    cut = 1+3*2*camCount
    focalLength   = vect[0]
    camera_Rts = np.reshape(vect[1:cut], (3, 2, camCount), order='F')
    worldPoints = np.reshape(vect[cut:], (imagePtsCount,3), order='F' )
    return focalLength, camera_Rts, worldPoints

def reprojectionResidual(imagePointIDs, imagePoints, px, py, focalLength, Camera_Rts, worldPoints):
    cameraCount = len(imagePointIDs[0])

    residuals = np.zeros((0, 0))
    
    for i in range(cameraCount):
        validPointID_IDs = imagePointIDs[:, i] != -1
        validImagePtIDs = imagePointIDs[validPointID_IDs, i]

        validCamRts = Camera_Rts[:, 0, i]
        validWorldPts = worldPoints[validPointID_IDs, :]

        RP = AngleAxisRotatePts(validCamRts, validWorldPts)

        TRX = RP[:,0] + Camera_Rts[0, 1, i]
        TRY = RP[:,1] + Camera_Rts[1, 1, i]
        TRZ = RP[:,2] + Camera_Rts[2, 1, i]

        TRXoZ = TRX / TRZ
        TRYoZ = TRY / TRZ

        x = focalLength * TRXoZ + px
        y = focalLength * TRYoZ + py

        ox = imagePoints[validImagePtIDs, 0]
        oy = imagePoints[validImagePtIDs, 1]

        step = np.vstack([(x-ox), (y-oy)])

        if i == 0:
            residuals = step
        else:
            residuals = np.hstack([residuals, step])

    return residuals.flatten()


def AngleAxisRotatePts(validCamRts, validWorldPts):
    validWorldPts=np.transpose(validWorldPts)
    angle_axis = np.reshape(validCamRts[0:3], (1,3))
    theta2 = np.inner(angle_axis, angle_axis)

    if theta2 > 0.0:
        theta = np.sqrt(theta2)
        w = (1.0/theta) * angle_axis

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        w_cross_pt = np.dot(vectorToSkewMat(w), validWorldPts)

        w_dot_pt = np.dot(w, validWorldPts)
        t1= (validWorldPts * costheta)
        t2 = (w_cross_pt * sintheta)
        t3 = np.dot((1 - costheta) * np.transpose(w), w_dot_pt)
        result = t1 + t2 + t3

    else:
        w_cross_pt = np.dot(vectorToSkewMat(angle_axis),validWorldPts)
        result = validWorldPts + w_cross_pt
        
    return np.transpose(result)

def vectorToSkewMat(a):
    assert(a.shape[0] == 1  and a.shape[1] ==3)
    ax=a[0,0]
    ay=a[0,1]
    az=a[0,2]
    A=np.array([[0, -az,ay],[az,0,-ax],[-ay,ax,0]])
    return A


def axisAngle2RotationMatrix(angle_axis):
    R= np.zeros((3,3))
    theta2 = np.inner(angle_axis, angle_axis)
    if theta2 > 0.0:
        theta = np.sqrt(theta2)
        wx = angle_axis[0] / theta
        wy = angle_axis[1] / theta
        wz = angle_axis[2] / theta
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        R[0,0] = costheta + wx * wx * (1 - costheta)
        R[1,0] = wz * sintheta + wx * wy * (1 - costheta)
        R[2,0] = -wy * sintheta + wx * wz * (1 - costheta)
        R[0,1] = wx * wy * (1 - costheta) - wz * sintheta
        R[1,1] = costheta + wy * wy * (1 - costheta)
        R[2,1] = wx * sintheta + wy * wz * (1 - costheta)
        R[0,2] = wy * sintheta + wx * wz * (1 - costheta)
        R[1,2] = -wx * sintheta + wy * wz * (1 - costheta)
        R[2,2] = costheta + wz * wz * (1 - costheta)
        
    else:
        R[0,0] = 1
        R[1,0] = -angle_axis[2]
        R[2,0] = angle_axis[1]
        R[0,1] = angle_axis[2]
        R[1,1] = 1
        R[2,1] = -angle_axis[0]
        R[0,2] = -angle_axis[1]
        R[1,2] = angle_axis[0]
        R[2,2] = 1
        
    return R


if __name__ == '__main__':
    pass


