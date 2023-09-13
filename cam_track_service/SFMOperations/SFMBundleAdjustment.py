import numpy as np
from cv2 import Rodrigues
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from SFMOperations.SFMDataTypes import MatchedImagePoint


# Tested residual weight threshold 0.292


class BundleAdjustment(object):
    def pack_two_view_baseline_parameters(self, t_scale: float, world_points: np.ndarray):
        # Calculate weights of parameters
        return np.insert(world_points.flatten(), 0, t_scale)

    def unpack_two_view_baseline_parameters(self, x: np.ndarray):
        l = x.shape[0]
        return x[0], x[1:].reshape(int((l - 1) / 3), 3)

    def two_view_baseline_residual(self, x: np.ndarray, image_point_0: np.ndarray, image_point_1: np.ndarray,
                                   P_0: np.ndarray, P_1: np.ndarray) -> float:

        t_scale, world_position = self.unpack_two_view_baseline_parameters(x)
        world_position = np.hstack([world_position,
                                    np.ones((world_position.shape[0], 1))]).reshape((world_position.shape[0], 4, 1))

        h_x_0 = np.matmul(P_0, world_position)
        h_x_1 = np.matmul(P_1, world_position)
        x_0 = h_x_0[:, [0, 1]] / h_x_0[:, [2]]
        x_1 = h_x_1[:, [0, 1]] / h_x_1[:, [2]]
        r_0 = image_point_0 - x_0.squeeze(axis=2)
        r_1 = image_point_1 - x_1.squeeze(axis=2)

        residual_2d = np.hstack([r_0, r_1])

        return residual_2d.flatten()

    def two_view_bundle_adjustment(self, baseline_distance: float, world_points: np.ndarray,
                                   image_point_0: np.ndarray,
                                   image_point_1: np.ndarray,
                                   P_0,
                                   P_1):

        x = self.pack_two_view_baseline_parameters(baseline_distance, world_points)
        residual_fun = self.two_view_baseline_residual
        solution = least_squares(residual_fun, x, method='lm', args=(image_point_0, image_point_1, P_0, P_1),
                                 max_nfev=1000)

        print(f'Two view projective error: {np.linalg.norm(solution.fun)}')
        t_scale, adjusted_world_points = self.unpack_two_view_baseline_parameters(solution.x)
        return t_scale, adjusted_world_points

    def two_view_ba_with_sparsity_matrix(self,
                                         camera_Rts: np.ndarray,
                                         world_pts: np.ndarray,
                                         image_pts: np.ndarray,
                                         k: np.ndarray,
                                         show_graphic=False):
        def rotate(world_pts, rvecs):
            theta = np.linalg.norm(rvecs, axis=1)[:, np.newaxis]
            with np.errstate(invalid='ignore'):
                v = rvecs / theta
                v = np.nan_to_num(v)
            the_dot = np.sum(world_pts * v, axis=1)[:, np.newaxis]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            return cos_theta * world_pts + sin_theta * np.cross(v, world_pts) + \
                   the_dot * (1 - cos_theta) * v

        def project(world_pts, camera_params, f):
            cam_world_pts = rotate(world_pts, camera_params[:, :3]) + camera_params[:, 3:6]
            project_points = -cam_world_pts[:, :2] / cam_world_pts[:, 2, np.newaxis]
            return f * project_points

        def project_back(world_pts, camera_params, k):
            homo_world_pts = np.hstack([world_pts, np.ones(shape=(world_pts.shape[0], 1), dtype=float)])
            projected_image_pts = np.empty((0, 3))
            for a_camera_parameter in camera_params:
                r_mat = Rodrigues(a_camera_parameter[:3])[0]
                tvec = a_camera_parameter[3:]
                Rt = np.hstack([r_mat, tvec.reshape((3, 1))])
                project_matrix = np.matmul(k, Rt)
                homo_image_pts = np.matmul(project_matrix, homo_world_pts.T)
                projected_image_pts = np.append(projected_image_pts, homo_image_pts.T, axis=0)

            return projected_image_pts[:, :2] / projected_image_pts[:, [2]]

        def fun(params, n_cameras, n_world_pts, image_pts, k):
            """

            :param params:
            :param n_cameras:
            :param n_world_pts:
            :param image_pts: n_cameras X n_world_pts X 2 in order
            :param camera_to_world_pts: n_cameras X n_world_pts [camera ID, world_pts ID]
            :return:
            """
            camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
            world_pts = params[n_cameras * 6:].reshape((n_world_pts, 3))
            project_image_points = project_back(world_pts, camera_params, k)
            return (image_pts - project_image_points).ravel()

        def fun_weighted(params, n_cameras, n_world_pts, image_pts, k):
            camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
            world_pts = params[n_cameras * 6:].reshape((n_world_pts, 3))
            project_image_points = project_back(world_pts, camera_params, k)
            raw_residuals = (image_pts - project_image_points).flatten()

            weights = np.ones(raw_residuals.shape) * 0.292
            std_deviation = np.std(raw_residuals)
            std_mean = np.mean(raw_residuals)
            weights[np.abs(raw_residuals - std_mean) < std_deviation * 0.292] = (2 - 0.292) ** 2

            return raw_residuals * weights

        def bundle_adjustment_sparsity(n_cameras, n_world_pts, world_pts_to_camera: np.ndarray):
            m = world_pts_to_camera.shape[0] * 2
            n = n_cameras * 6 + n_world_pts * 3
            A = lil_matrix((m, n), dtype=int)
            i = np.arange(world_pts_to_camera.shape[0])
            for j in range(6):
                A[2 * i, world_pts_to_camera[:, 1] * 6 + j] = 1
                A[2 * i + 1, world_pts_to_camera[:, 1] * 6 + j] = 1

            for j in range(3):
                A[2 * i, n_cameras * 6 + world_pts_to_camera[:, 0] * 3 + j] = 1
                A[2 * i + 1, n_cameras * 6 + world_pts_to_camera[:, 0] * 3 + j] = 1

            return A

        def decomposite_solution(x: np.ndarray, n_cameras: int, n_world_pts: int):
            camera_indices = np.arange(n_cameras)
            world_pts_indices = np.arange(n_world_pts)
            cam_rvecs = np.empty((n_cameras, 3), dtype=float)
            cam_tvecs = np.empty((n_cameras, 3), dtype=float)
            world_pts = np.empty((n_world_pts, 3), dtype=float)

            for i in range(3):
                cam_rvecs[camera_indices, i] = x[camera_indices * 6 + i]
                cam_tvecs[camera_indices, i] = x[camera_indices * 6 + 3 + i]
                world_pts[world_pts_indices, i] = x[n_cameras * 6 + world_pts_indices * 3 + i]

            return (cam_rvecs, cam_tvecs, world_pts)

        n_cameras = camera_Rts.shape[0]
        n_world_pts = world_pts.shape[0]
        X_0 = np.hstack([camera_Rts.ravel(), world_pts.ravel()])

        world_pts_to_camera = np.empty((n_cameras * n_world_pts, 2), dtype=int)
        world_pts_to_camera[:, 0] = np.tile(np.arange(n_world_pts), n_cameras)
        world_pts_to_camera[:, 1] = np.repeat(np.arange(n_cameras), n_world_pts)

        f_0 = fun(X_0, n_cameras, n_world_pts, image_pts, k)
        f_0_weighted = fun_weighted(X_0, n_cameras, n_world_pts, image_pts, k)

        A = bundle_adjustment_sparsity(n_cameras, n_world_pts, world_pts_to_camera)
        result = least_squares(fun, X_0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                               args=(n_cameras, n_world_pts, image_pts, k))

        result_weighted = least_squares(fun_weighted, X_0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4,
                                        method='trf', args=(n_cameras, n_world_pts, image_pts, k))

        if show_graphic:
            fig, axes = plt.subplots(4, 1, sharex=True)
            fig.suptitle("Residual of Back Projection")
            axes[0].plot(f_0)
            axes[0].grid(True)
            axes[0].set_title("Before BA")
            axes[1].plot(result.fun)
            axes[1].grid(True)
            axes[1].set_title("After BA")

            axes[2].plot(f_0_weighted)
            axes[2].grid(True)
            axes[2].set_title("Weighted before BA")
            axes[3].plot(result_weighted.fun)
            axes[3].grid(True)
            axes[3].set_title("Weighted after BA")

            fig.tight_layout(pad=0.5)
            win = plt.get_current_fig_manager()
            win.set_window_title("two_view_ba_with_sparsity_matrix")
            win.resize(1000, 1000)
            plt.show()
        # return decomposite_solution(result.x, n_cameras, n_world_pts)

        return decomposite_solution(result.x, n_cameras, n_world_pts), \
               decomposite_solution(result_weighted.x, n_cameras, n_world_pts)

    def two_view_ba_with_sparsity_matrix_wieght_update(self,
                                                       camera_Rts: np.ndarray,
                                                       world_pts: np.ndarray,
                                                       image_pts: np.ndarray,
                                                       k: np.ndarray,
                                                       weight_update: float):
        def project_back(world_pts, camera_params, k):
            homo_world_pts = np.hstack([world_pts, np.ones(shape=(world_pts.shape[0], 1), dtype=float)])
            projected_image_pts = np.empty((0, 3))
            for a_camera_parameter in camera_params:
                r_mat = Rodrigues(a_camera_parameter[:3])[0]
                tvec = a_camera_parameter[3:]
                Rt = np.hstack([r_mat, tvec.reshape((3, 1))])
                project_matrix = np.matmul(k, Rt)
                homo_image_pts = np.matmul(project_matrix, homo_world_pts.T)
                projected_image_pts = np.append(projected_image_pts, homo_image_pts.T, axis=0)
            return projected_image_pts[:, :2] / projected_image_pts[:, [2]]

        def fun_weighted(params, n_cameras, n_world_pts, image_pts, k, weight_update):
            camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
            world_pts = params[n_cameras * 6:].reshape((n_world_pts, 3))
            project_image_points = project_back(world_pts, camera_params, k)
            raw_residuals = (image_pts - project_image_points).flatten()
            weights = np.ones(raw_residuals.shape) * weight_update
            std_deviation = np.std(raw_residuals)
            std_mean = np.mean(raw_residuals)
            weights[np.abs(raw_residuals - std_mean) < std_deviation * weight_update] = (2 - weight_update) ** 2
            return raw_residuals * weights

        def bundle_adjustment_sparsity(n_cameras, n_world_pts, world_pts_to_camera: np.ndarray):
            m = world_pts_to_camera.shape[0] * 2
            n = n_cameras * 6 + n_world_pts * 3
            A = lil_matrix((m, n), dtype=int)
            i = np.arange(world_pts_to_camera.shape[0])
            for j in range(6):
                A[2 * i, world_pts_to_camera[:, 1] * 6 + j] = 1
                A[2 * i + 1, world_pts_to_camera[:, 1] * 6 + j] = 1

            for j in range(3):
                A[2 * i, n_cameras * 6 + world_pts_to_camera[:, 0] * 3 + j] = 1
                A[2 * i + 1, n_cameras * 6 + world_pts_to_camera[:, 0] * 3 + j] = 1

            return A

        def decomposite_solution(x: np.ndarray, n_cameras: int, n_world_pts: int):
            camera_indices = np.arange(n_cameras)
            world_pts_indices = np.arange(n_world_pts)
            cam_rvecs = np.empty((n_cameras, 3), dtype=float)
            cam_tvecs = np.empty((n_cameras, 3), dtype=float)
            world_pts = np.empty((n_world_pts, 3), dtype=float)

            for i in range(3):
                cam_rvecs[camera_indices, i] = x[camera_indices * 6 + i]
                cam_tvecs[camera_indices, i] = x[camera_indices * 6 + 3 + i]
                world_pts[world_pts_indices, i] = x[n_cameras * 6 + world_pts_indices * 3 + i]

            return (cam_rvecs, cam_tvecs, world_pts)

        n_cameras = camera_Rts.shape[0]
        n_world_pts = world_pts.shape[0]
        X_0 = np.hstack([camera_Rts.ravel(), world_pts.ravel()])

        world_pts_to_camera = np.empty((n_cameras * n_world_pts, 2), dtype=int)
        world_pts_to_camera[:, 0] = np.tile(np.arange(n_world_pts), n_cameras)
        world_pts_to_camera[:, 1] = np.repeat(np.arange(n_cameras), n_world_pts)
        A = bundle_adjustment_sparsity(n_cameras, n_world_pts, world_pts_to_camera)

        result_weighted = least_squares(fun_weighted, X_0, jac_sparsity=A, verbose=0, x_scale='jac', ftol=1e-4,
                                        method='trf', args=(n_cameras, n_world_pts, image_pts, k, weight_update))

        return decomposite_solution(result_weighted.x, n_cameras, n_world_pts)

    def ba_on_camera_paramter(self, camera_Rts: np.ndarray, true_world_pts: np.ndarray, true_image_pts: np.ndarray,
                              k: np.ndarray, graphic_debug=False):
        assert true_world_pts.shape[0] * camera_Rts.shape[0] == true_image_pts.shape[0]

        def rotate(world_pts, rvecs):
            theta = np.linalg.norm(rvecs, axis=1)[:, np.newaxis]
            with np.errstate(invalid='ignore'):
                v = rvecs / theta
                v = np.nan_to_num(v)
            the_dot = np.sum(world_pts * v, axis=1)[:, np.newaxis]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            return cos_theta * world_pts + sin_theta * np.cross(v, world_pts) + \
                   the_dot * (1 - cos_theta) * v

        def project(world_pts, camera_params, f):
            cam_world_pts = rotate(world_pts, camera_params[:, :3]) + camera_params[:, 3:6]
            project_points = -cam_world_pts[:, :2] / cam_world_pts[:, 2, np.newaxis]
            return f * project_points

        def project_back(world_pts, camera_params, k):
            homo_world_pts = np.hstack([world_pts, np.ones(shape=(world_pts.shape[0], 1), dtype=float)])
            projected_image_pts = np.empty((0, 3))
            for a_camera_parameter in camera_params:
                r_mat = Rodrigues(a_camera_parameter[:3])[0]
                tvec = a_camera_parameter[3:]
                Rt = np.hstack([r_mat, tvec.reshape((3, 1))])
                project_matrix = np.matmul(k, Rt)
                homo_image_pts = np.matmul(project_matrix, homo_world_pts.T)
                projected_image_pts = np.append(projected_image_pts, homo_image_pts.T, axis=0)

            return projected_image_pts[:, :2] / projected_image_pts[:, [2]]

        def fun(params, n_cameras, world_pts, image_pts, k: np.ndarray):
            camera_params = params.reshape((n_cameras, 6))
            project_image_points = project_back(world_pts, camera_params, k)
            return (image_pts - project_image_points).ravel()

        def fun_weighted(params, n_cameras, world_pts, image_pts, k: np.ndarray):
            camera_params = params.reshape((n_cameras, 6))
            project_image_points = project_back(world_pts, camera_params, k)
            raw_residuals = (image_pts - project_image_points).flatten()
            weights = np.ones(raw_residuals.shape) * 0.382
            std_deviation = np.std(raw_residuals)
            std_mean = np.mean(raw_residuals)
            weights[np.abs(raw_residuals - std_mean) < std_deviation * 0.382] = 1.618 * 2
            # weights = np.exp(std_deviation / np.abs(raw_residuals - std_mean))
            return raw_residuals * weights

        n_cameras = camera_Rts.shape[0]
        X_0 = camera_Rts.ravel()

        f_0 = fun(X_0, n_cameras, true_world_pts, true_image_pts, k)
        f_0_weighted = fun_weighted(X_0, n_cameras, true_world_pts, true_image_pts, k)

        result = least_squares(fun, X_0, method='lm', ftol=1e-5, args=(n_cameras, true_world_pts, true_image_pts, k))
        result_weighted = least_squares(fun_weighted, X_0, method='lm', ftol=1e-5,
                                        args=(n_cameras, true_world_pts, true_image_pts, k))

        if graphic_debug:
            fig, axes = plt.subplots(4, 1, sharex=True)
            fig.suptitle("Residual of Back Projection")
            axes[0].plot(f_0)
            axes[0].grid(True)
            axes[0].set_title("Before BA")
            axes[1].plot(result.fun)
            axes[1].grid(True)
            axes[1].set_title("After BA")

            axes[2].plot(f_0_weighted)
            axes[2].grid(True)
            axes[2].set_title("Weighted before BA")
            axes[3].plot(result_weighted.fun)
            axes[3].grid(True)
            axes[3].set_title("Weighted after BA")

            fig.tight_layout(pad=3.0)
            win = plt.get_current_fig_manager()
            win.set_window_title("ba_on_camera_paramter")
            win.resize(1000, 800)
            plt.show()

        return result.x.reshape(camera_Rts.shape), result_weighted.x.reshape(camera_Rts.shape)

    def ba_on_camera_paramter_weighted(self, camera_Rts: np.ndarray, true_world_pts: np.ndarray,
                                       true_image_pts: np.ndarray, camera_matrix: np.ndarray,
                                       residual_weight_threshold: float):

        def project_back(world_pts, camera_params, k):
            homo_world_pts = np.hstack([world_pts, np.ones(shape=(world_pts.shape[0], 1), dtype=float)])
            projected_image_pts = np.empty((0, 3))
            for a_camera_parameter in camera_params:
                r_mat = Rodrigues(a_camera_parameter[:3])[0]
                tvec = a_camera_parameter[3:]
                Rt = np.hstack([r_mat, tvec.reshape((3, 1))])
                project_matrix = np.matmul(k, Rt)
                homo_image_pts = np.matmul(project_matrix, homo_world_pts.T)
                projected_image_pts = np.append(projected_image_pts, homo_image_pts.T, axis=0)
            return projected_image_pts[:, :2] / projected_image_pts[:, [2]]

        def fun_weighted(params, camera_count, world_pts, image_pts, k: np.ndarray, weight_threshold: float):
            camera_params = params.reshape((camera_count, 6))
            project_image_points = project_back(world_pts, camera_params, k)

            raw_residuals = (image_pts - project_image_points).flatten()

            weights = np.ones(raw_residuals.shape) * weight_threshold
            std_deviation = np.std(raw_residuals)
            std_mean = np.mean(raw_residuals)
            weights[np.abs(raw_residuals - std_mean) < std_deviation * weight_threshold] = (2 - weight_threshold) ** 2
            return raw_residuals * weights

        assert true_world_pts.shape[0] * camera_Rts.shape[0] == true_image_pts.shape[0]

        n_cameras = camera_Rts.shape[0]
        X_0 = camera_Rts.ravel()
        result_weighted = least_squares(fun_weighted, X_0, method='lm', ftol=1e-5,
                                        args=(n_cameras, true_world_pts, true_image_pts, camera_matrix,
                                              residual_weight_threshold))
        return result_weighted.x.reshape(camera_Rts.shape)

    @staticmethod
    def n_view_ba_sparsity_weighted_resid(camera_rts: np.ndarray, world_pts: np.ndarray, image_pts: np.ndarray,
                                          cam_intrinsic: np.ndarray, weight_update: float):

        def project_back(world_pt_set, camera_params, k):
            homo_world_pts = np.hstack([world_pt_set, np.ones(shape=(world_pt_set.shape[0], 1), dtype=float)])
            projected_image_pts = np.empty((0, 3))
            for a_camera_parameter in camera_params:
                r_mat = Rodrigues(a_camera_parameter[:3])[0]
                tvec = a_camera_parameter[3:]
                rt_mat = np.hstack([r_mat, tvec.reshape((3, 1))])
                project_matrix = np.matmul(k, rt_mat)
                homo_image_pts = np.matmul(project_matrix, homo_world_pts.T)
                projected_image_pts = np.append(projected_image_pts, homo_image_pts.T, axis=0)

            return projected_image_pts[:, :2] / projected_image_pts[:, [2]]

        def fun_weighted(params, cameras_count, world_pt_count, image_pt_set, k, residual_weight):
            camera_params = params[: cameras_count * 6].reshape((cameras_count, 6))
            world_points = params[cameras_count * 6:].reshape((world_pt_count, 3))
            project_image_points = project_back(world_points, camera_params, k)
            raw_residuals = (image_pt_set - project_image_points).flatten()

            weights = np.ones(raw_residuals.shape) * residual_weight
            std_deviation = np.std(raw_residuals)
            std_mean = np.mean(raw_residuals)
            weights[np.abs(raw_residuals - std_mean) < std_deviation * residual_weight] = (2 - residual_weight) ** 2

            return raw_residuals * weights

        def bundle_adjustment_sparsity(camera_count, world_pt_count, world_pts_cameras_map: np.ndarray):
            m = world_pts_cameras_map.shape[0] * 2
            n = camera_count * 6 + world_pt_count * 3
            A = lil_matrix((m, n), dtype=int)
            i = np.arange(world_pts_cameras_map.shape[0])
            for j in range(6):
                A[2 * i, world_pts_cameras_map[:, 1] * 6 + j] = 1
                A[2 * i + 1, world_pts_cameras_map[:, 1] * 6 + j] = 1

            for j in range(3):
                A[2 * i, camera_count * 6 + world_pts_cameras_map[:, 0] * 3 + j] = 1
                A[2 * i + 1, camera_count * 6 + world_pts_cameras_map[:, 0] * 3 + j] = 1

            return A

        def decompose_solution(x: np.ndarray, cam_count: int, world_pt_count: int):
            camera_indices = np.arange(cam_count)
            world_pts_indices = np.arange(world_pt_count)
            cam_rvecs = np.empty((cam_count, 3), dtype=float)
            cam_tvecs = np.empty((cam_count, 3), dtype=float)
            world_pts = np.empty((world_pt_count, 3), dtype=float)

            for i in range(3):
                cam_rvecs[camera_indices, i] = x[camera_indices * 6 + i]
                cam_tvecs[camera_indices, i] = x[camera_indices * 6 + 3 + i]
                world_pts[world_pts_indices, i] = x[cam_count * 6 + world_pts_indices * 3 + i]
            return cam_rvecs, cam_tvecs, world_pts

        n_cameras = camera_rts.shape[0]
        n_world_pts = world_pts.shape[0]
        X_0 = np.hstack([camera_rts.ravel(), world_pts.ravel()])

        world_pts_to_camera = np.empty((n_cameras * n_world_pts, 2), dtype=int)
        world_pts_to_camera[:, 0] = np.tile(np.arange(n_world_pts), n_cameras)
        world_pts_to_camera[:, 1] = np.repeat(np.arange(n_cameras), n_world_pts)

        sparsity_mat = bundle_adjustment_sparsity(n_cameras, n_world_pts, world_pts_to_camera)
        result_weighted = least_squares(fun_weighted, X_0, jac_sparsity=sparsity_mat, verbose=0, x_scale='jac',
                                        ftol=1e-4,
                                        method='trf',
                                        args=(n_cameras, n_world_pts, image_pts, cam_intrinsic, weight_update))

        return decompose_solution(result_weighted.x, n_cameras, n_world_pts)

    @staticmethod
    def n_view_ba_sparsity_weighted_data_remapped(camera_intrinsic: np.ndarray, camera_parameter_list: np.ndarray,
                                                  world_point_list: np.ndarray,
                                                  image_point_list: np.ndarray,
                                                  id_map_camera_parameter_world_point_image_point: np.ndarray):

        def bundle_adjustment_sparsity(cam_param_width: int, cam_count: int, world_pt_count: int,
                                       id_map_cam_param_world_pt_image_pt: np.ndarray):

            m = id_map_cam_param_world_pt_image_pt.shape[0] * 2
            n = cam_count * cam_param_width + world_pt_count * 3
            sparsity_mat = lil_matrix((m, n), dtype=int)
            image_pt_idx = np.arange(id_map_cam_param_world_pt_image_pt.shape[0])
            for param_offset in range(cam_param_width):
                sparsity_mat[2 * image_pt_idx, id_map_cam_param_world_pt_image_pt[:,
                                               0] * cam_param_width + param_offset] = 1
                sparsity_mat[2 * image_pt_idx + 1, id_map_cam_param_world_pt_image_pt[:,
                                                   0] * cam_param_width + param_offset] = 1
            for pos_offset in range(3):
                sparsity_mat[
                    2 * image_pt_idx, cam_count * cam_param_width + id_map_cam_param_world_pt_image_pt[
                                                                        :, 1] * 3 + pos_offset] = 1
                sparsity_mat[
                    2 * image_pt_idx + 1, cam_count * cam_param_width + id_map_cam_param_world_pt_image_pt[
                                                                            :, 1] * 3 + pos_offset] = 1
            return sparsity_mat

        def project_back(cam_params: np.ndarray, world_pts: np.ndarray,
                         id_map_cam_param_world_pt_image_pt: np.ndarray):
            projected_2d = np.zeros((id_map_cam_param_world_pt_image_pt.shape[0], 2))
            for cam_param_id, world_pt_id, image_pt_id in id_map_cam_param_world_pt_image_pt:
                rvec, tvec = cam_params[cam_param_id, :3], cam_params[cam_param_id, 3:]
                r_mat = Rodrigues(rvec)[0]
                rt_to_scene = np.hstack((r_mat, tvec.reshape(3, 1)))
                project_mat = np.matmul(camera_intrinsic, rt_to_scene)
                h_world_pos = np.hstack((world_pts[world_pt_id], 1))
                h_image_2d = np.matmul(project_mat, h_world_pos)
                projected_2d[image_pt_id] = h_image_2d[:2] / h_image_2d[2]
            return projected_2d

        def fun(bundle_params: np.ndarray, cam_count: int, cam_param_width: int, world_pt_count: int,
                image_points: np.ndarray, id_map_cam_param_world_pt_image_pt: np.ndarray):
            camera_param_list = bundle_params[: cam_count * cam_param_width].reshape((cam_count, cam_param_width))
            world_pts_list = bundle_params[cam_count * cam_param_width:].reshape((world_pt_count, 3))
            residuals = image_points - project_back(camera_param_list, world_pts_list,
                                                    id_map_cam_param_world_pt_image_pt)
            return residuals.ravel()

        def decompose_solution(x: np.ndarray, cam_count: int, cam_param_width: int, world_pt_count: int):
            camera_indices = np.arange(cam_count)
            world_pts_indices = np.arange(world_pt_count)
            cam_rvecs = np.empty((cam_count, 3), dtype=float)
            cam_tvecs = np.empty((cam_count, 3), dtype=float)
            world_pts = np.empty((world_pt_count, 3), dtype=float)

            for i in range(3):
                cam_rvecs[camera_indices, i] = x[camera_indices * cam_param_width + i]
                cam_tvecs[camera_indices, i] = x[camera_indices * cam_param_width + 3 + i]
                world_pts[world_pts_indices, i] = x[cam_count * cam_param_width + world_pts_indices * 3 + i]
            return cam_rvecs, cam_tvecs, world_pts

        initial_values = np.hstack((camera_parameter_list.ravel(), world_point_list.ravel()))
        camera_count = camera_parameter_list.shape[0]
        camera_parameter_width = camera_parameter_list.shape[1]
        world_point_count = world_point_list.shape[0]
        ba_sparsity_mat = bundle_adjustment_sparsity(camera_parameter_width, camera_count, world_point_count,
                                                     id_map_camera_parameter_world_point_image_point)

        ba_result = least_squares(fun, initial_values, jac_sparsity=ba_sparsity_mat, verbose=0, x_scale='jac',
                                  ftol=1e-4, method='trf',
                                  args=(camera_count, camera_parameter_width, world_point_count, image_point_list,
                                        id_map_camera_parameter_world_point_image_point))

        return decompose_solution(ba_result.x, camera_count, camera_parameter_width, world_point_count)

    @staticmethod
    def n_view_data_remapped_residual(camera_parameter_list: np.ndarray, world_point_list: np.ndarray,
                                      image_point_list: np.ndarray, camera_intrinsic: np.ndarray,
                                      id_map_camera_parameter_world_point_image_point: np.ndarray):

        def project_back(cam_params: np.ndarray, world_pts: np.ndarray,
                         id_map_cam_param_world_pt_image_pt: np.ndarray):
            projected_2d = np.zeros((id_map_cam_param_world_pt_image_pt.shape[0], 2))
            for cam_param_id, world_pt_id, image_pt_id in id_map_cam_param_world_pt_image_pt:
                rvec, tvec = cam_params[cam_param_id, :3], cam_params[cam_param_id, 3:]
                r_mat = Rodrigues(rvec)[0]
                rt_to_scene = np.hstack((r_mat, tvec.reshape(3, 1)))
                project_mat = np.matmul(camera_intrinsic, rt_to_scene)
                h_world_pos = np.hstack((world_pts[world_pt_id], 1))
                h_image_2d = np.matmul(project_mat, h_world_pos)
                projected_2d[image_pt_id] = h_image_2d[:2] / h_image_2d[2]
            return projected_2d

        def image_residual(camera_param_list: np.ndarray, world_pts_list: np.ndarray, image_points: np.ndarray,
                           id_map_cam_param_world_pt_image_pt: np.ndarray):
            residuals = image_points - project_back(camera_param_list, world_pts_list,
                                                    id_map_cam_param_world_pt_image_pt)
            std_deviation = np.std(residuals)
            std_mean = np.mean(residuals)

            return std_mean, std_deviation

        mean, deviation = image_residual(camera_parameter_list, world_point_list, image_point_list,
                                         id_map_camera_parameter_world_point_image_point)
        return mean, deviation


    def n_view_residual(self,
                        camera_Rts: np.ndarray,
                        world_pts: np.ndarray,
                        image_pts: np.ndarray,
                        k: np.ndarray,
                        global_residual=True
                        ):
        def project_back(the_world_pts, camera_params, k):
            homo_world_pts = np.hstack([the_world_pts, np.ones(shape=(the_world_pts.shape[0], 1), dtype=float)])
            projected_image_pts = np.empty((0, 3))
            for a_camera_parameter in camera_params:
                r_mat = Rodrigues(a_camera_parameter[:3])[0]
                tvec = a_camera_parameter[3:]
                Rt = np.hstack([r_mat, tvec.reshape((3, 1))])
                project_matrix = np.matmul(k, Rt)
                homo_image_pts = np.matmul(project_matrix, homo_world_pts.T)
                projected_image_pts = np.append(projected_image_pts, homo_image_pts.T, axis=0)

            return projected_image_pts[:, :2] / projected_image_pts[:, [2]]

        def image_residual(params, cameras_count, world_pts_count, raw_image_pts, k):
            camera_params = params[: cameras_count * 6].reshape((cameras_count, 6))
            the_world_pts = params[cameras_count * 6:].reshape((world_pts_count, 3))
            project_image_points = project_back(the_world_pts, camera_params, k)
            raw_residuals = (raw_image_pts - project_image_points).flatten()
            std_deviation = np.std(raw_residuals)
            std_mean = np.mean(raw_residuals)
            return std_mean, std_deviation

        def per_image_residual(params, cameras_count, world_pts_count, raw_image_pts, k):
            camera_params = params[: cameras_count * 6].reshape((cameras_count, 6))
            the_world_pts = params[cameras_count * 6:].reshape((world_pts_count, 3))
            project_image_points = project_back(the_world_pts, camera_params, k)
            raw_residuals = (raw_image_pts - project_image_points).flatten()
            per_image_raw_residuals = raw_residuals.reshape(cameras_count, world_pts_count * 2)
            std_deviation = np.std(per_image_raw_residuals, axis=1)
            std_mean = np.mean(per_image_raw_residuals, axis=1)
            return std_mean, std_deviation

        n_cameras = camera_Rts.shape[0]
        n_world_pts = world_pts.shape[0]
        world_pts_to_camera = np.empty((n_cameras * n_world_pts, 2), dtype=int)
        world_pts_to_camera[:, 0] = np.tile(np.arange(n_world_pts), n_cameras)
        world_pts_to_camera[:, 1] = np.repeat(np.arange(n_cameras), n_world_pts)
        X_0 = np.hstack([camera_Rts.ravel(), world_pts.ravel()])
        if global_residual:
            return image_residual(X_0, n_cameras, n_world_pts, image_pts, k)
        else:
            return per_image_residual(X_0, n_cameras, n_world_pts, image_pts, k)
