import numpy as np
from SFMOperations.FundamentalMatrixCalculations import RANSAC_FM
from SFMOperations.vgg_Functions import vgg_X_from_xP_nonlin

class SFMOperators(object):
    
    def __init__(self):
        self.FM_Algorithm = RANSAC_FM
        self.focalLength = 1.0
        self.pointsA = None
        self.pointsB = None
        self.imageSize = None
        self.intrinsicMatrix = None
        
    def setCorrespondences(self, pointsA, pointsB):
        self.pointsA = pointsA
        self.pointsB = pointsB
    
    def setFocalLength(self, inFocalLength):
        self.focalLength = inFocalLength
        
    def setImageSize(self, inImageSize):
        self.imageSize = inImageSize
        
    def getImageSize(self):
        return self.imageSize
        
        
    def initialCameraIntrinsic(self):
        intrinsicMatrix = np.eye(3)
        intrinsicMatrix[0][0] = self.focalLength
        intrinsicMatrix[1][1] = self.focalLength
        self.intrinsicMatrix = intrinsicMatrix
        return intrinsicMatrix
        
    
    def getFundamentalMatrix(self):
        if self.pointsA is not None and self.pointsA.shape == self.pointsB.shape:
            
            FM_Data = self.FM_Algorithm(self.pointsA, self.pointsB)
            FM = FM_Data[0]
            if FM.shape == (3, 3):
                return FM
            else:
                print("Debug: Fundamental matrix calculation failure!")
                return None
            
        else:
            print("Debug: Points data does not initialed yet!")
            return None
        
        
    def getEM_From_FM(self, F, K):
        if isinstance(F, np.ndarray) and isinstance(K, np.ndarray):            
            return np.dot(np.transpose(K), np.dot(F, K))
        else:
            print("Debug: Input data type error!")
            return None
        
    def getCameraPose(self):
        if self.imageSize is None:
            print("Debug: Image Size is not initialed yet!")
            return None
        
        if self.pointsA is None or self.pointsB is None:
            print("Debug: Correspondences data is not initialed yet!")
            return None            
        
        F = self.getFundamentalMatrix()
        K = self.intrinsicMatrix
        E = self.getEM_From_FM(F, K)
        
        if K is None:
            print("Initial intrinsic matrix first please!")
            return None
        
        Rt = self.estimateCameraPose(E, K, np.hstack([self.pointsA, self.pointsB]), self.imageSize)
        
        return Rt
    
    
    def estimateCameraPose(self, E, K, matches, imageSize):
        n, d = matches.shape
    
        U, s, V = np.linalg.svd(E, full_matrices=True)
        W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
        Z = np.array([[0,1,0], [-1,0,0], [0,0,0]])
    
        S = np.dot(np.dot(U,Z), np.transpose(U))
    
        R1 = np.dot(np.dot(U,W), V)
        R2 = np.dot(np.dot(U, np.transpose(W)), V)
    
        t1 = U[:,2]
        t2 = -1*U[:,2]
    
        if np.linalg.det(R1) < 0:
            print("Negative determinant F1 multiplied by -1")
            R1 = -1 * R1
    
        if np.linalg.det(R2) < 0:
            print("Negative determinant R2 multiplied by -1")
            R2 = -1 * R2
    
        # This generates 4 possible solutions
        t1t = t1.reshape(3,1)
        t2t = t2.reshape(3,1)
        sols = [np.hstack((R1, t1t)), np.hstack((R1, t2t)), np.hstack((R2, t1t)), np.hstack((R2, t2t))]
    
        Rt = np.zeros((3,4,4))
        Rt[:,:,0] = sols[0]
        Rt[:, :, 1] = sols[1]
        Rt[:, :, 2] = sols[2]
        Rt[:, :, 3] = sols[3]
    
        # For each solution
        P0 = np.dot(K,np.hstack([np.eye(3),np.zeros((3,1))]))
        goodV = np.zeros((1,4))
        
        for i in range(4):
            outX = np.zeros((n, 4))
            P1 = np.dot(K,sols[i])
    
            # For every pair of 2D points
            for j in range(n):
    
                # apply vgg to calculate 3D points
                colIms = np.array([imageSize[1],imageSize[0]]).reshape((2,1))
                imsize = colIms.repeat(2,axis=1)
                pt = np.zeros((P0.shape[0],P0.shape[1],2))
                pt[:, :, 0] = P0
                pt[:, :, 1] = P1
                formatedMatched = np.reshape(matches[j,:],(2, 2))
                outX[j,:] = vgg_X_from_xP_nonlin(formatedMatched, pt, imsize)
    
            # Apply scale
            outX = outX[:,0:3] / outX[:,[3,3,3]]
    
            t = Rt[0:3, 3, i].reshape((3, 1))
            aux = np.transpose(outX[:,:]) - np.repeat(t, outX.shape[0], axis=1)
            t2 = Rt[2, 0:3, i].reshape((1, 3))
            dprd = np.dot(t2, aux)
            goodV[0,i] = np.sum([np.bitwise_and(outX[:, 2] > 0, dprd > 0)])
            
        # Calculate which is better
        bestIndex = np.argmax(goodV)
        
        return Rt[:, :, bestIndex]
    
    
    def getFullIntrinsicMatrix(self):
        matrixToSave = self.intrinsicMatrix.copy()
        matrixToSave[0, 2] = self.imageSize[0] / 2
        matrixToSave[1, 2] = self.imageSize[1] / 2
        
        return matrixToSave
        
    
    