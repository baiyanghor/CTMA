import cv2
import os
import numpy as np
import maya.api.OpenMayaUI as OMUI2
import maya.api.OpenMaya as OM2
import maya.cmds as MCMDs

imageFileForlder = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment/UnitTest/openCVAPITest/test-image/"
fixCameraName = 'SequenceTestCamera'

def CV_Reconstruction():
    fileNameList = os.listdir(imageFileForlder)
    fileList = [imageFileForlder + aFileName for aFileName in fileNameList]
    print(fileList)
    sl = OM2.MGlobal.getSelectionListByName(fixCameraName)

    if sl.length() > 0:
        dp = sl.getDagPath(0)
        fn_CamDN = OM2.MFnDagNode(dp)
        camera_transform = OM2.MTransformationMatrix(fn_CamDN.transformationMatrix())

        camera_center_vector = camera_transform.translation(OM2.MSpace.kWorld)
        camera_rotation_vector = camera_transform.rotation(False).asVector()
        camera_center = np.array([camera_center_vector.x, camera_center_vector.y, camera_center_vector.z])
        rotation_angle_radians = np.array([camera_rotation_vector.x, camera_rotation_vector.y, camera_rotation_vector.z])
        
        if fn_CamDN.childCount() > 0:
            fn_camShapeNode = fn_CamDN.child(0)
            
            if fn_camShapeNode.apiType() == OM2.MFn.kCamera:
                fn_Cam = OM2.MFnCamera(fn_camShapeNode)
                camera_aperture_in_mm = fn_Cam.horizontalFilmAperture * 25.4
                f_maya = fn_Cam.focalLength

                maya_projectMatrix = fn_Cam.projectionMatrix()

        size_X = MCMDs.getAttr('defaultResolution.width')
        size_Y = MCMDs.getAttr('defaultResolution.height')


        P, K, R, t = getProjectMatrixFromMaya(f_maya, size_X, size_Y, rotation_angle_radians, camera_center, camera_aperture_in_mm)        
        # print("OpenCV Project Matrix")
        # print(P)
        # print("Maya Project Matrix")
        # print(maya_projectMatrix)
        # print(K)

        print("P")
        print(P)
        print("K")
        print(K)
        
        # print(cv2.sfm.reconstruct(images = fileList, K = K, is_projective = True))
    else:
        return None



def getProjectMatrixFromMaya(f_maya, size_X, size_Y, rotation_angle_radians, camera_center, camera_aperture_in_mm):
    px = size_X / 2.0
    py = size_Y / 2.0

    f = (f_maya / camera_aperture_in_mm) * size_X
    print("OpenCV Focal Length")
    print(f)
    K = np.eye(3)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = px
    K[1, 2] = py

    rotation_angle_radians = -1 * rotation_angle_radians

    R = eul2rotm_xyz(rotation_angle_radians)

    t = (-1 * R).dot(np.transpose(camera_center))

    F = np.eye(3)
    F[1, 1] = -1
    F[2, 2] = -1

    t = F.dot(t)
    R = F.dot(R)

    t = t.reshape((3, 1))
    
    Rt = np.concatenate((R, t), axis=1)

    P = K.dot(Rt)

    return P, K, R, t

def eul2rotm_xyz(rotation_angle_radians):
    R_x = np.eye(3)
    R_x[1, 1] = np.cos(rotation_angle_radians[0])
    R_x[1, 2] = -1 * np.sin(rotation_angle_radians[0])
    R_x[2, 1] = np.sin(rotation_angle_radians[0])
    R_x[2, 2] = np.cos(rotation_angle_radians[0])

    R_y = np.eye(3)
    R_y[0, 0] = np.cos(rotation_angle_radians[1])
    R_y[0, 2] = np.sin(rotation_angle_radians[1])
    R_y[2, 0] = -1 * np.sin(rotation_angle_radians[1])
    R_y[2, 2] = np.cos(rotation_angle_radians[1])

    R_z = np.eye(3)
    R_z[0, 0] = np.cos(rotation_angle_radians[2])
    R_z[0, 1] = -1 * np.sin(rotation_angle_radians[2])
    R_z[1, 0] = np.sin(rotation_angle_radians[2])
    R_z[1, 1] = np.cos(rotation_angle_radians[2])

    R = R_x.dot(R_y.dot(R_z))

    return R
