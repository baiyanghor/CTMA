import numpy as np
from scipy import linalg


def formatVgg(X):
    if len(X) != 3:
        return None

    row1 = [0,     X[2],  -X[1]]
    row2 = [-X[2], 0,     X[0]]
    row3 = [X[1],  -X[0], 0]

    return np.array([row1, row2, row3])


def resid(Y, u, Q):
    K = Q.shape[2]

    q = Q[:, 0:3, 0]
    x0 = Q[:, 3, 0]
    x0 = x0.reshape((3, 1))
    x = np.dot(q, Y) + x0

    tu = u[0,:].reshape((2,1))
    e = x[0:2]/x[2]-tu

    t1 = x[2]* q[0, :]
    t2 = x[0]* q[2, :]
    t3 = x[2]* q[1, :]
    t4 = x[1]* q[2, :]
    aux = np.vstack([t1 - t2, t3 - t4])

    J = (aux / (x[2] * x[2]))

    for i in range(1, K):
        q = Q[:, 0:3, i]
        x0 = Q[:, 3, i]
        x0 = x0.reshape((3, 1))
        x = np.dot(q, Y) + x0
        tu = u[i, :].reshape((2, 1))
        e = np.vstack([e, x[0:2]/x[2]-tu])

        t1 = x[2] * q[0, :]
        t2 = x[0] * q[2, :]
        t3 = x[2] * q[1, :]
        t4 = x[1] * q[2, :]

        aux = np.vstack([t1-t2, t3-t4])
        J = np.vstack([J, (aux/(x[2]*x[2]))])

    return e, J


def vgg_X_from_xP_lin(u, P, imsize):
    K = P.shape[2]
    newu = u.copy()
    newP = P.copy()
    if not imsize is None:
        for i in range(K):
            H = np.array([[2.0/imsize[0, i], 0,                -1],
                          [0,                2.0/imsize[1, i], -1],
                          [0,                0,                 1]])
            newP[:,:,i] = np.dot(H, newP[:, :, i])
            newu[i, :] = np.dot(H[0:2,0:2], newu[i, :]) + H[0:2, 2]

    A= np.dot( formatVgg(np.hstack([newu[0,:],1])),  newP[:, :, 0])
    for i in range(1, K):
        newRow = np.dot(formatVgg(np.hstack([newu[i, :], 1])), newP[:, :, i])
        A = np.vstack([A, newRow])

    _, _, out = np.linalg.svd(A)
    out = out.T
    out = out[:, -1]
    s = np.dot(np.reshape(newP[2, :, :], (4, K)).T, out)
    if np.any(s < 0):
        out = -out

    return out


def vgg_X_from_xP_nonlin(u, P, imsize=None, X=None):
    eps = 2.2204e-16

    K = P.shape[2]
    assert(K >= 2)
    
    # First I get X if it was not providing
    if X is None:
        X = vgg_X_from_xP_lin(u, P, imsize)

    # Fixed the -1 ????? (in old version the indices differ from a number)
    # Maybe for faster convergence Byang. Hor
    newu = u.copy() - 1
    # newu = u.copy()
    newP = P.copy()

    if not imsize is None:
        for i in range(K):
            H = np.array([[2.0/imsize[0,i], 0,               -1],
                          [0,               2.0/imsize[1,i], -1],
                          [0,               0,                1]
                          ])
            newP[:, :, i] = np.dot(H, newP[:, :, i])
            newu[i, :] = np.dot(H[0:2, 0:2], newu[i, :]) + H[0:2, 2]
    # Why SVD here ?
    T, s, U = np.linalg.svd(X.reshape((4, 1)))
    lc = T.shape[1]
    T = T[:, [1, 2, 3, 0]]
    Q = newP.copy()
    for i in range(K):
        Q[:, :, i] = np.dot(newP[:, :, i], T )

    # DO THE NEWTON
    # Why do newton again? Byang. Hor
    Y = np.zeros((3,1))
    eprev = np.inf

    for i in range(10):
        e, j = resid(Y, newu, Q)
        if (1 - np.linalg.norm(e) / np.linalg.norm(eprev)) < 1000 * eps:
            break

        eprev = e
        jj = np.dot(np.transpose(j), j)
        je = np.dot(np.transpose(j), e)
        # Y  = Y - np.linalg.solve(jj, je)
        Y = Y - np.dot(linalg.pinv(jj), je)

    X = np.dot(T, np.vstack([Y, 1]))

    return X.flatten()

