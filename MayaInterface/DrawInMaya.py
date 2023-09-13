import maya.api.OpenMaya as OM2
import maya.cmds as MCMD
import numpy as np
from ImportRawSFM import SFMDataManipulator

class DrawSFMData(object):

    def __init__(self, aSFMDataManupilator = SFMDataManipulator()):
        self.dataManipulator = aSFMDataManupilator
        
    
    def npMatrixToMMatrix(self, npMatrix):
        elements = np.hstack((npMatrix.flatten(), np.array([0,0,0,1])))
        return OM2.MMatrix(elements.tolist())
    

    def drawWorldPoints(self):
        worldPoints = self.dataManipulator.getWorldPoints()
        worldPoints[1] = -1 * worldPoints[1]
        worldPoints[2] = -1 * worldPoints[2]
        MCMD.nParticle(position = worldPoints.tolist())
        

    def drawCameras(self):
        cameraRts = self.dataManipulator.getCameraRt()
        intrinsicMatrix = self.dataManipulator.getIntrinsicMatrix()

        tranformMatrix = np.eye(4, dtype=np.float)
        for i in range(cameraRts.shape[2]):

            tranformMatrix = np.dot( np.vstack((cameraRts[:,:,i], np.array([0,0,0,1]))), tranformMatrix)
            camera_aperture_inch, size_X, size_Y, focalLength, q_rotation, camera_center = self.openCVCameraToMaya(intrinsicMatrix,
                                                                                             tranformMatrix[0:3, 0:3],
                                                                                             tranformMatrix[0:3, 3],
                                                                                             camera_aperture_in_mm = 37.0)

            camera_fn = OM2.MFnCamera()
            camera_fn.create()
            camera_fn.focalLength = focalLength
            camera_fn.setHorizontalFieldOfView(camera_aperture_inch)
            cameraTransform_fn = OM2.MFnTransform(camera_fn.parent(0))
            cameraTransform_fn.setName("Frame_00" + str(i + 1))

            cameraTranslation = OM2.MVector(tuple(camera_center.flatten().tolist()))  
            
            cameraTransform_fn.translateBy(cameraTranslation, OM2.MSpace.kTransform)

            if OM2.MVector(q_rotation.asAxisAngle()[0]).length() == 0.0:
                q_rotation.y = 1.0


            cameraTransform_fn.setRotation(q_rotation, OM2.MSpace.kObject)
            
    
    def rotateMatrixToQuaternion(self, rotateMatrix = np.array([], dtype=np.float)):
        eta = 0.0
        
        pseudoTraces = np.array([rotateMatrix[0,0] + rotateMatrix[1,1] + rotateMatrix[2,2],
                        rotateMatrix[0,0] - rotateMatrix[1,1] - rotateMatrix[2,2],
                        -rotateMatrix[0,0] + rotateMatrix[1,1] - rotateMatrix[2,2],
                        -rotateMatrix[0,0] - rotateMatrix[1,1] + rotateMatrix[2,2]])

        pluses = np.array([rotateMatrix[0,1] + rotateMatrix[1,0],
                            rotateMatrix[0,2] + rotateMatrix[2,0],
                            rotateMatrix[1,2] + rotateMatrix[2,1]])

        minuses = np.array([rotateMatrix[2,1] - rotateMatrix[1,2],
                            rotateMatrix[0,2] - rotateMatrix[2,0],
                            rotateMatrix[1,0] - rotateMatrix[0,1]
                            ])
        
        squaredNumerators = np.array([
                            np.square(minuses[0]) + np.square(minuses[1]) + np.square(minuses[2]),
                            np.square(minuses[0]) + np.square(pluses[0]) + np.square(pluses[1]),
                            np.square(minuses[1]) + np.square(pluses[0]) + np.square(pluses[2]),
                            np.square(minuses[2]) + np.square(pluses[1]) + np.square(pluses[2])
                            ])
        
        q = np.array([0, 0, 0, 0], dtype=np.float)

        for i in range(4):
            if pseudoTraces[i] > eta:
                q[i] = np.sqrt(1 + pseudoTraces[i])
            else:
                q[i] = np.sqrt(squaredNumerators[i]/(3 - pseudoTraces[i]))
        
        q[1:] = q[1:] * np.sign(minuses)

        q = q[[1,2,3,0]]/2
        q[1] = -1.0 * q[1]
        # q[[1,2]] = -1 * q[[1,2]]

        return OM2.MQuaternion(q.tolist())


    def openCVCameraToMaya(self, intrinsicMatrix = np.array([]), R = np.array([]), t = np.array([]), camera_aperture_in_mm = 36.0):
        size_X = intrinsicMatrix[0, 2] * 2.0
        size_Y = intrinsicMatrix[1, 2] * 2.0

        f_avg = (intrinsicMatrix[0,0] + intrinsicMatrix[1,1]) / 2.0
        f_maya = (f_avg * camera_aperture_in_mm) / size_X

        F = np.eye(3)
        F[1, 1] = -1
        F[2, 2] = -1

        
        t = np.dot(F, t)
        R = np.dot(F, R)

        camera_center = -1.0 * np.dot(R.T, t)
        camera_center[0] = -1.0 * camera_center[0]
        R = R.T

        rotation = self.rotateMatrixToQuaternion(R)

        camera_aperture_in_inch = camera_aperture_in_mm / 25.4
        
        return camera_aperture_in_inch, size_X, size_Y, f_maya, rotation, camera_center

