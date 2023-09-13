import cv2
import numpy as np

def openCV_FM(pointsA, pointsB):
    fundamentalMatrix, mask = cv2.findFundamentalMat(pointsA, pointsB, cv2.FM_RANSAC, 1.0, 0.99)
    return fundamentalMatrix, mask


def RANSAC_FM(pointsA, pointsB, minDistance=0.002):
    print(f"Total points count {pointsA.shape[0]}, {pointsB.shape[0]}")
    # Distance at which it is considered outlier
    F, inliers = RANSAC_Fit_FM(pointsA, pointsB, minDistance)
    print('RANSAC_FM')
    print("Inliners {0}".format(len(inliers)))

    print('INLIERS percentage {0}'.format(len(inliers)*1.0 / len(pointsA)))
    print("----------------------------------")
    return F, inliers

def RANSAC_Fit_FM(pointsA, pointsB, tolerance):
    assert(pointsA.shape == pointsB.shape)

    # Normalize so that the origin is cnetroid and mean distance from the origin is sqrt (2)
    # It also ensures that the scale parameter is 1
    na, Ta = normalizeHomogeneous(pointsA)
    nb, Tb = normalizeHomogeneous(pointsB)

    # Points for estimating the fundamental matrix
    s = 8
    # Send to RANSAC algorithm (get model with more inliners)
    modelFunction = DLT_FM
    distanceFunction = distanceModel
    isDegenerate = lambda x: False
    # Nothing is degenerate

    # Add to the hstack in each row x1, x2 3 + 3 6 elements per row

    dataset = np.hstack([na,nb])
    inliners, M = ransac(dataset, modelFunction, distanceFunction, isDegenerate, s, tolerance)

    F = DLT_FM(np.hstack([na[inliners,:], nb[inliners, :]]))

    F = np.dot(np.dot(Tb, F), np.transpose(Ta))

    return F, inliners


def DLT_FM(data):
    assert(data.shape[1] == 6 )

    p1,p2 = data[:,0:3],data[:,3:]
    n, d = p1.shape

    na, Ta = normalizeHomogeneous(p1)
    nb, Tb = normalizeHomogeneous(p2)

    p2x1p1x1 = nb[:,0] * na[:,0]
    p2x1p1x2 = nb[:,0] * na[:,1]
    p2x1 = nb[:, 0]
    p2x2p1x1 = nb[:,1] * na[:,0]
    p2x2p1x2 = nb[:,1] * na[:,1]
    p2x2 = nb[:,1]
    p1x1 = na[:,0]
    p1x2 = na[:,1]
    ones = np.ones((1,p1.shape[0]))

    A = np.vstack([p2x1p1x1, p2x1p1x2, p2x1, p2x2p1x1, p2x2p1x2, p2x2, p1x1, p1x2, ones])
    A = np.transpose(A)

    u, D, v = np.linalg.svd(A)
    vt = v.T

    F = vt[:, 8].reshape(3,3) 
    # Get the vector with the lowest eigenvalue and that is F
    # Since the fundamental matrix is rank 2, we must redo svd and rebuild
    # From rank 2

    u, D, v = np.linalg.svd(F)
    F = np.dot(np.dot(u, np.diag([D[0], D[1], 0])), v)

    F = np.dot(np.dot(Tb, F), np.transpose(Ta))

    return F

def distanceModel(F, x, t):
    p1, p2 = x[:, 0:3], x[:, 3:]

    x2tFx1 = np.zeros((p1.shape[0],1))

    x2ftx1 = [np.dot(np.dot(p2[i], F), np.transpose(p1[i])) for i in range(p1.shape[0])]

    ft1 = np.dot(F, np.transpose(p1))
    ft2 = np.dot(F.T, np.transpose(p2))

    sumSquared = (np.power(ft1[0, :], 2) +
                  np.power(ft1[1, :], 2)) + \
                 (np.power(ft2[0, :], 2) +
                  np.power(ft2[1, :], 2))
    
    d34 = np.power(x2ftx1, 2) / sumSquared
    bestInliers = np.where(np.abs(d34) < t)[0]
    bestF = F
    
    return bestInliers, bestF


def ransac(x, fittingfn, distfn, degenfn, s, t):
    maxTrials = 2000
    maxDataTrials = 200
    p = 0.99

    bestM = None
    trialCount = 0
    maxInlinersYet = 0
    N = 1
    maxN = 120
    n, d = x.shape

    M = None
    bestInliners = None
    
    while N > trialCount:
        degenerate = 1
        degenerateCount = 1
        while degenerate:
            inds = np.random.choice(range(n), s, replace=False)
            sample = x[inds,:]
            degenerate = degenfn(sample)

            if not degenerate:
                M = fittingfn(sample)
                if M is None:
                    degenerate = 1
            degenerateCount += 1
            if degenerateCount > maxDataTrials:
                # raise Exception("Error muchas sample degeneradas saliendo")
                raise Exception("Error many degenerate samples coming out")


        # Evaluate model
        inliners, M = distfn(M,x,t)
        nInliners = len(inliners)

        if maxInlinersYet < nInliners:
            maxInlinersYet = nInliners
            bestM = M
            bestInliners = inliners
            
            # Probability estimation trials until obtain
            eps = 0.000001
            fractIn = nInliners*1.0 / n
            pNoOut = 1 - fractIn*fractIn
            pNoOut = max(eps, pNoOut)
            # Avoid division by 0
            N = np.log(1-p) / np.log(pNoOut)
            N = max(N, maxN)

        trialCount += 1
        if trialCount > maxTrials:

            print("Maximum iteration was reached exiting")
            break
        
    if M is None:
        raise Exception("Model was not found error")

    print("Realization {0} Attempts".format(trialCount))

    return bestInliners, bestM




def normalizeHomogeneous(points):
    if points.shape[1] == 2:
        # I add scale factor (concatenate column with 1)
        points = np.hstack([points, np.ones((points.shape[0],1))])

    n = points.shape[0]
    d = points.shape[1]

    # Leaves on scale 1
    factores = np.repeat((points[:, -1].reshape(n, 1)), d, axis=1)
    points = points / factores

    # NOTE THAT THIS IS BY ELEMENT
    prom = np.mean(points[:,:-1],axis=0)
    newP = np.zeros(points.shape)

    # Leave all dimensions on average 0 (minus scale)
    newP[:,:-1] = points[:,:-1] - np.vstack([prom for i in range(n)])

    # Calculate average distance
    dist = np.sqrt(np.sum(np.power(newP[:,:-1],2),axis=1))
    meanDis = np.mean(dist)
    scale = np.sqrt(2)*1.0/ meanDis

    T = [[scale, 0,     -scale*prom[0]],
         [0,     scale, -scale*prom[1]],
         [0,     0,     1             ]
         ]

    # THIS IS THE ORIGINAL VERSION THAT WAS USED T * points
    # It assumes DxN points as points are used in NxD format
    # The transpose is used
    
    T = np.transpose(np.array(T))
    transformedPoints = np.dot(points,T)

    return transformedPoints, T
