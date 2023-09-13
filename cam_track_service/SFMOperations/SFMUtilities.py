import numpy as np
import cv2 as cv
from collections import OrderedDict, defaultdict

def skew_vector(vec: np.ndarray) -> np.ndarray:
    mat = np.array([[0      , vec[2], -vec[1]],
                    [-vec[2], 0     ,  vec[0]],
                    [vec[1] , -vec[0], 0     ]
                    ])
    return mat


def homo_vector(vec: np.ndarray) -> np.ndarray:
    h_vector = np.asarray(vec.flatten().tolist() + [1.0])
    return h_vector


def homo_h_vector_list(vec_list: np.ndarray) -> np.ndarray:
    append_ones = np.ones((vec_list.shape[0], 1))
    h_vector_list = np.hstack([vec_list, append_ones])
    return h_vector_list


def normalized_p_0() -> np.ndarray:
    return np.hstack([np.eye(3), np.array([[0], [0], [0]])])


def get_ordered_image_ids_selection(image_id_range: list, ordered_merge_operation_ids: list,
                                    merged_image_id_dict: dict) -> dict:
    image_id_selection_dict = {}
    rest = set(image_id_range)
    for op_id in ordered_merge_operation_ids:
        if rest.issuperset(merged_image_id_dict[op_id]):
            image_id_selection_dict.setdefault(op_id, merged_image_id_dict[op_id])
            rest = rest - set(merged_image_id_dict[op_id])
        elif rest.intersection(set(merged_image_id_dict[op_id])):
            image_id_selection_dict.setdefault(op_id, rest.intersection(set(merged_image_id_dict[op_id])))
            rest = rest - rest.intersection(set(merged_image_id_dict[op_id]))
        elif set(merged_image_id_dict[op_id]).issuperset(rest):
            image_id_selection_dict.setdefault(op_id, list(rest))

    return image_id_selection_dict


def ordered_merge_ops_rt_to_scene(ops_rt_dict: dict, ordered_op_ids: list, required_image_range: list):
    ordered_rt_dict = OrderedDict()
    for op_id in ordered_op_ids:
        ordered_rt_dict[op_id] = ops_rt_dict[op_id]

    rest_image_ids = set(map(str, required_image_range))
    select_rts = defaultdict(list)
    for op_id, rts_dict in ordered_rt_dict.items():
        image_ids = set(rts_dict.keys())
        if intersect_ids := image_ids & rest_image_ids:
            rest_image_ids -= intersect_ids
            select_rts[op_id].extend(sorted(intersect_ids, key=lambda v: int(v)))
        else:
            continue

    return select_rts


def get_shifted_range(ordered_list: list, affected_range: list, original_range:list) -> list:
    if affected_range[0] >= original_range[0]:
        start_ind = original_range.index(affected_range[0])
    else:
        raise ValueError("Required image ID out of range!")
    if affected_range[-1] <= original_range[-1]:
        end_ind = original_range.index(affected_range[-1])
    else:
        raise ValueError("Required image ID out of range!")

    return ordered_list[start_ind: end_ind + 1]


def rt_matrix_for_ba(rt_matrix: np.ndarray) -> np.ndarray:
    assert rt_matrix.shape == (3, 4)
    rvec = cv.Rodrigues(rt_matrix[:, :3])[0].flatten()
    tvec = rt_matrix[:, [3]].flatten()
    return np.hstack([rvec, tvec])


def rt_to_scene_from_vec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r_mat = cv.Rodrigues(rvec)[0]
    return np.hstack([r_mat, tvec.reshape(3, 1)])


def intrinsic_from_image_info(focal_length: float, image_size: tuple) -> np.ndarray:
    camera_matrix = np.array([[focal_length, 0, image_size[1] / 2.0],
                              [0, focal_length, image_size[0] / 2.0],
                              [0, 0, 1]])
    return camera_matrix

