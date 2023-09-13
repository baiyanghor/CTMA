import maya.cmds as MCMD
import os
import numpy as np

class SFMDataManipulator(object):
    
    def __init__(self):
        self.worldPointsFileName = None
        self.cameraRtsFileName = None
        self.intrinsicMatrixFileName = None

    def setFilePaths(self, cameraRtsFileName, worldPtsfileName, intrinsicMatrixFileName):
        if os.path.isfile(worldPtsfileName) and os.path.isfile(cameraRtsFileName):
            self.cameraRtsFileName = cameraRtsFileName
            self.worldPointsFileName = worldPtsfileName
            self.intrinsicMatrixFileName = intrinsicMatrixFileName
        else:
            print "Debug: Camera Rt or world points data file doesn't exists!"

    def getWorldPoints(self):
        return self.__loadFromTextFile(self.worldPointsFileName)


    def getCameraRt(self):
        return self.__loadFromTextFile(self.cameraRtsFileName)

    
    def getIntrinsicMatrix(self):
        return self.__loadFromTextFile(self.intrinsicMatrixFileName)


    def __loadFromTextFile(self, fileName = ''):
        shapeFileName =os.path.join(os.path.dirname(fileName), os.path.basename(fileName).split('.')[0] + "_shape.txt")

        if os.path.isfile(fileName) and os.path.isfile(shapeFileName):

            data = np.loadtxt(fileName)

            with open(shapeFileName, 'r') as readHandler:
                aLine = readHandler.readline()
            
            dataShape = tuple([int(x.strip()) for x in aLine.split(',')])

            return np.reshape(data, dataShape, order='C')

        else:
            print "Debug: SFM data file doesn't exists!"
            return None


