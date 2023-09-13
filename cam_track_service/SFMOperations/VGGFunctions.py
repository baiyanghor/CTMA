"""
Functions referencing Visual Geometry Group
Author: Byang Hor
"""

"""
Notations:
x for data from image points
X for points in world coordinate frame
"""
__author__ = 'Byang Hor'
import numpy as np
from SFMOperations.SFMUtilities import homo_vector, skew_vector

def geo_residual(Y: np.ndarray, x: np.ndarray, Q: np.ndarray):
    """
    :param Y: 3 x 1 ndarray
    :param x: k x 2   normalized 2d point positions ndarray
    :param Q: k x 3 x 4 normalize project ndarray
    :return: e: 2*k x 2  , J 2*k x 3
    """

    point2d_count = x.shape[0]
    e = np.zeros((point2d_count,2))
    J = np.zeros((point2d_count, 2, 3))

    for i in range(point2d_count):
        q = Q[i, :, 0:3]
        X_0 = Q[i, :, 3, np.newaxis]
        X = np.matmul(q, Y) + X_0

        e[i] = (X[0:2] / X[2, 0]).reshape((1,2)) - x[i]
        J_i_0 = (np.dot(X[2], q[[0], :]) - np.dot(X[0], q[[2], :])) / X[2, 0] ** 2
        J_i_1 = (np.dot(X[2], q[[1], :]) - np.dot(X[1], q[[2], :])) / X[2, 0] ** 2
        J[i,:,:] = np.vstack([J_i_0, J_i_1])

    return e.reshape((point2d_count*2, 1)), J.reshape((point2d_count*2, 3))


def normalize_x_P(pts_position_2d: np.ndarray, project_matrices: np.ndarray, image_sizes: tuple):
    assert pts_position_2d.shape[0] == project_matrices.shape[0]

    point_2d_count = pts_position_2d.shape[0]

    normalized_position = np.zeros(pts_position_2d.shape)
    normalized_project_matrices = np.zeros(project_matrices.shape)

    # Normalize translation matrix
    H = np.array([[2.0/image_sizes[1], 0,                  -1],
                  [0,                  2.0/image_sizes[0], -1],
                  [0,                  0,                   1]
                  ])

    for i in range(point_2d_count):
        normalized_position[i] = np.matmul(H[0:2, 0:2], pts_position_2d[i]) + H[0:2, 2]
        normalized_project_matrices[i] = np.matmul(H, project_matrices[i])

    return normalized_position, normalized_project_matrices

def X_from_xP_linear(pts_position_2d: np.ndarray, project_matrices: np.ndarray, image_sizes: tuple):
    """
    For in house usage all image size are same in on calculation session
    :param pts_position_2d: k x 2 ndarray
    :param project_matrices: k x 3 x 4 ndarray
    :param image_sizes: k x 2 ndarray
    :return: 1 x 4 ndarray
    """
    assert len(pts_position_2d) == len(project_matrices)
    point_2d_count = len(pts_position_2d)

    normalized_position, normalized_project_matrices = normalize_x_P(pts_position_2d, project_matrices, image_sizes)

    A = np.dot(skew_vector(homo_vector(normalized_position[0])), normalized_project_matrices[0])
    for i in range(1, point_2d_count):
        new_row = np.matmul(skew_vector(homo_vector(normalized_position[i])), normalized_project_matrices[i])
        A = np.vstack([A, new_row])

    _, _, Vh = np.linalg.svd(A, full_matrices=True)
    candidate = Vh[[-1], :]
    sign_check = np.matmul(normalized_project_matrices[:, 2, :], candidate.T)

    if np.any(sign_check < 0):
        candidate = -candidate

    return candidate


def X_from_xP_nonlinear(pts_position_2d: np.ndarray, project_matrices: np.ndarray, image_sizes: tuple, initial_X = None):
    """
    For in house usage all image size are same in on calculation session
    :param pts_position_2d: k x 2 ndarray
    :param project_matrices: k x 3 x 4 ndarray
    :param image_sizes: tuple(height, width)
    :return: 1 x 4 ndarray
    """

    assert len(pts_position_2d) == len(project_matrices)
    point_2d_count = len(pts_position_2d)

    if initial_X is None:
        guess_X = X_from_xP_linear(pts_position_2d, project_matrices, image_sizes)
    else:
        guess_X = initial_X

    normalized_position, normalized_project_matrices = normalize_x_P(pts_position_2d, project_matrices, image_sizes)
    # Parametrize X such that X = T*[Y;1]; thus x = P*T*[Y;1] = Q*[Y;1]
    _, _, Vh = np.linalg.svd(guess_X.reshape((1,4)))
    T = Vh[:, [1, 2, 3, 0]]
    Q = np.zeros(normalized_project_matrices.shape)
    for i in range(point_2d_count):
        Q[i] = np.matmul(normalized_project_matrices[i], T)

    Y = np.zeros((3, 1))

    eprev = np.inf
    epsilon = np.finfo(float).eps
    # Found solution with newton methods
    for i in range(10):
        e, J = geo_residual(Y, normalized_position, Q)
        if 1-np.linalg.norm(e)/np.linalg.norm(eprev) < 1000 * epsilon:
            break
        eprev = e
        Y = Y - np.matmul(np.linalg.pinv(np.matmul(J.T, J)), np.matmul(J.T, e))

    homo_world_point_3d = np.matmul(T, homo_vector(Y).reshape(4, 1))

    return homo_world_point_3d/homo_world_point_3d[3, 0]
