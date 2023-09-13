import maya.api.OpenMaya as OM2
import numpy as np

pi = np.pi
sn = np.sin(pi/4)
cs = np.cos(pi/4)
rotateXwith_45 = [1,0,0,0, 0,cs,-sn,0, 0,sn,cs,0, 0,0,0,1]
rotateTransform = OM2.MTransformationMatrix(OM2.MMatrix(rotateXwith_45))
camera_fn = OM2.MFnCamera()
camera = camera_fn.create()

cameraTransform_fn = OM2.MFnTransform(camera_fn.parent(0))
cameraTransform_fn.setName("Frame_001")
cameraTransform_fn.setTransformation(rotateTransform)