import cv2 as cv
import os
import numpy as np
import shutil
import math


class OpenCVOperators(object):
    def __init__(self):
        ...

    def bf_matcher(self, descriptors_0, descriptors_1):
        # Ensure that desA, desB are in np float 32 format
        descriptors_0 = descriptors_0.astype(np.float32)
        descriptors_1 = descriptors_1.astype(np.float32)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors_0, descriptors_1, k=2)

        # Apply ratio test
        selected_indices_0 = []
        selected_indices_1 = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                selected_indices_0.append(m.queryIdx)
                selected_indices_1.append(m.trainIdx)

        return selected_indices_0, selected_indices_1

    def twoview_match_sift(self, raw_mat_0, raw_mat_1, show=False):

        matchmaker = self.bf_matcher

        feature_extractor = cv.SIFT_create(contrastThreshold=0.04)
        mat_gray_0 = cv.cvtColor(raw_mat_0, cv.COLOR_BGR2GRAY)
        mat_gray_1 = cv.cvtColor(raw_mat_1, cv.COLOR_BGR2GRAY)

        keypoints_0, descriptors_0 = feature_extractor.detectAndCompute(mat_gray_0, None)
        keypoints_1, descriptors_1 = feature_extractor.detectAndCompute(mat_gray_1, None)

        if show:
            img_mat_0 = cv.drawKeypoints(descriptors_0, keypoints_0, None)
            img_mat_1 = cv.drawKeypoints(descriptors_1, keypoints_1, None)
            cv.imshow('Image_0', img_mat_0)
            cv.imshow('Image_1', img_mat_1)
            cv.waitKey()

        match_indices_0, match_indices_1 = matchmaker(descriptors_0, descriptors_1)

        if isinstance(keypoints_0[0], cv.KeyPoint):
            kp_position_0 = cv.KeyPoint_convert(keypoints_0)
            kp_position_1 = cv.KeyPoint_convert(keypoints_1)
        else:
            kp_position_0 = keypoints_0
            kp_position_1 = keypoints_1

        if show:
            self.draw_matches(mat_gray_0, kp_position_0, mat_gray_1, kp_position_1, match_indices_0, match_indices_1)

        matched_pos_0 = kp_position_0[np.array(match_indices_0)]
        matched_pos_1 = kp_position_1[np.array(match_indices_1)]

        return matched_pos_0, matched_pos_1

    def twoview_match_of(self, raw_mat_0, raw_mat_1):
        gray_mat_0 = cv.cvtColor(raw_mat_0, cv.COLOR_BGR2GRAY)
        gray_mat_1 = cv.cvtColor(raw_mat_1, cv.COLOR_BGR2GRAY)

        kp_position_0 = cv.goodFeaturesToTrack(image=gray_mat_0, maxCorners=256, qualityLevel=0.01, minDistance=10,
                                               blockSize=5, useHarrisDetector=False)
        kp_position_1 = cv.goodFeaturesToTrack(image=gray_mat_1, maxCorners=256, qualityLevel=0.01, minDistance=10,
                                               blockSize=5, useHarrisDetector=False)

        kp_position_0 = np.squeeze(kp_position_0, axis=1)
        kp_position_1 = np.squeeze(kp_position_1, axis=1)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        of_matched_in_1, of_status, err_distance = cv.calcOpticalFlowPyrLK(gray_mat_0, gray_mat_1, kp_position_0,
                                                                           None, **lk_params)

        idx_backup_matched_in_image_0 = list()
        found_points_in_image_1 = list()
        for i in range(len(of_matched_in_1)):
            if of_status[i] and err_distance[i] < 4.0:
                idx_backup_matched_in_image_0.append(i)
                found_points_in_image_1.append(of_matched_in_1[i])

        matcher = cv.BFMatcher_create(normType=cv.NORM_L2)

        np_found_points_in_image_1 = np.array(found_points_in_image_1)

        matched_pair_in_image_1 = matcher.radiusMatch(np_found_points_in_image_1, kp_position_1, 2)

        filtered_matches = list()
        repeat_check_list = list()

        for a_pair in matched_pair_in_image_1:
            temp_match = cv.DMatch()
            if len(a_pair) == 1:
                temp_match = a_pair[0]
            elif len(a_pair) > 1:
                if (a_pair[0].distance / a_pair[1].distance) < 0.7:
                    temp_match = a_pair[0]
                else:
                    continue
            else:
                continue

            if temp_match.trainIdx not in repeat_check_list:
                temp_match.queryIdx = idx_backup_matched_in_image_0[temp_match.queryIdx]
                filtered_matches.append(temp_match)
                repeat_check_list.append(temp_match.trainIdx)

        filtered_points_0 = list()
        filtered_points_1 = list()

        for a_pair in filtered_matches:
            filtered_points_0.append(kp_position_0[a_pair.queryIdx])
            filtered_points_1.append(kp_position_1[a_pair.trainIdx])

        return np.array(filtered_points_0), np.array(filtered_points_1)

    def twoview_match_orb_of(self, raw_mat_0: np.ndarray, raw_mat_1: np.ndarray) -> (np.ndarray, np.ndarray):
        gray_mat_0 = cv.cvtColor(raw_mat_0, cv.COLOR_BGR2GRAY)
        gray_mat_1 = cv.cvtColor(raw_mat_1, cv.COLOR_BGR2GRAY)

        feature_extractor = cv.ORB_create(scaleFactor=2,
                                          nlevels=3,
                                          edgeThreshold=15,
                                          WTA_K=2,
                                          patchSize=15,
                                          fastThreshold=8
                                          )

        kps_0 = feature_extractor.detect(gray_mat_0)
        kps_1 = feature_extractor.detect(gray_mat_1)

        kp_position_0 = cv.KeyPoint_convert(kps_0)
        kp_position_1 = cv.KeyPoint_convert(kps_1)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        of_matched_in_1, of_status, err_distance = cv.calcOpticalFlowPyrLK(gray_mat_0, gray_mat_1, kp_position_0,
                                                                           None, **lk_params)

        idx_backup_matched_in_image_0 = list()
        found_points_in_image_1 = list()

        for i in range(len(of_matched_in_1)):
            if of_status[i] and err_distance[i] < 4.0:
                idx_backup_matched_in_image_0.append(i)
                found_points_in_image_1.append(of_matched_in_1[i])

        matcher = cv.BFMatcher_create(normType=cv.NORM_L2)

        np_found_points_in_image_1 = np.array(found_points_in_image_1)
        matched_pair_in_image_1 = matcher.radiusMatch(np_found_points_in_image_1, kp_position_1, 2)

        filtered_matches = list()
        repeat_check_list = list()

        for a_pair in matched_pair_in_image_1:
            temp_match = cv.DMatch()
            if len(a_pair) == 1:
                temp_match = a_pair[0]
            elif len(a_pair) > 1:
                if (a_pair[0].distance / a_pair[1].distance) < 0.7:
                    temp_match = a_pair[0]
                else:
                    continue
            else:
                continue

            if temp_match.trainIdx not in repeat_check_list:
                temp_match.queryIdx = idx_backup_matched_in_image_0[temp_match.queryIdx]
                filtered_matches.append(temp_match)
                repeat_check_list.append(temp_match.trainIdx)

        filtered_points_0 = list()
        filtered_points_1 = list()

        for a_pair in filtered_matches:
            filtered_points_0.append(kp_position_0[a_pair.queryIdx])
            filtered_points_1.append(kp_position_1[a_pair.trainIdx])

        return np.array(filtered_points_0), np.array(filtered_points_1)

    def construct_mat_sequence(self, i_image_full_name_list: list):
        mat_list = []
        for a_image_file in i_image_full_name_list:
            if os.path.isfile(a_image_file):
                mat_list.append(cv.imread(a_image_file))

        if len(mat_list) == len(i_image_full_name_list):
            print("Debug: Get data from " + str(len(mat_list)) + " images!")
            return mat_list
        else:
            print("Debug: Required image files are not intact!")
            return None

    def twoview_get_matched_features(self, i_raw_mat_0: np.ndarray, i_raw_mat_1: np.ndarray,
                                     extractor_name='ORB_OF') -> (np.ndarray, np.ndarray):
        matchmaker = self.twoview_match_orb_of
        if extractor_name == 'SIFT':
            matchmaker = self.twoview_match_sift
        elif extractor_name == 'OF':
            matchmaker = self.twoview_match_of

        kp_position_0, kp_position_1 = matchmaker(i_raw_mat_0, i_raw_mat_1)

        return kp_position_0, kp_position_1

    def cv_rt_to_matrix(self, rvec, tvec):
        r_mat = cv.Rodrigues(rvec)[0]
        return np.hstack([r_mat, tvec.reshape(3, 1)])

    def generate_shadow_mask(self, src_image: np.ndarray):
        blur_image = cv.bilateralFilter(src_image, 5, 75, 75)

        # HSI CONVERSION
        blur_image = np.divide(blur_image, 255.0)
        B = blur_image[:, :, 0]
        G = blur_image[:, :, 1]
        R = blur_image[:, :, 2]

        I = np.mean(blur_image, axis=2)
        numerator = (R - G) + (R - B)
        dominator = 2 * np.sqrt(np.power((R - G), 2) + (R - B) * (G - B))
        theta = np.arccos(numerator / dominator)
        H = np.where(B < G, 2 * np.pi - theta, theta)
        ratio_map = H / (I + 0.00001)

        # OTSU'S METHOD
        ret, threshold_image = cv.threshold(ratio_map.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        dilate_kernel = np.ones((3, 3), np.uint8)
        shadow_mask = cv.dilate(threshold_image, dilate_kernel, iterations=3)
        return shadow_mask

    def generate_shadow_mask_list(self, shadow_mask_path, mat_list):
        def pad_file_name(padding: int, count: int):
            padded_str = str(int(math.pow(10, padding) + count))
            return padded_str[-padding:]

        if os.path.isdir(shadow_mask_path):
            shutil.rmtree(shadow_mask_path)

        os.mkdir(shadow_mask_path)

        mask_image_list = []
        for i, a_image_data in enumerate(mat_list):
            shadow_mask = self.generate_shadow_mask(a_image_data)
            save_file_name = os.path.join(shadow_mask_path, f'shadow_mask_{pad_file_name(4, i + 1)}.png')
            mask_image_list.append(save_file_name)
            cv.imwrite(save_file_name, shadow_mask)
        return mask_image_list

    @staticmethod
    def get_mask_image_filenames(shadow_mask_path):
        mask_image_list = []
        if os.path.isdir(shadow_mask_path):
            base_names = os.listdir(shadow_mask_path)
            base_names = sorted(base_names, reverse=False)
            for a_base_name in base_names:
                mask_image_list.append(os.path.join(shadow_mask_path, a_base_name))
            return mask_image_list

        else:
            return None

    @staticmethod
    def optical_track_features(features_to_track: np.ndarray, reference_features, im_mat_0: np.ndarray,
                               im_mat_1: np.ndarray):
        lk_parameters = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
            flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS
        )
        features_in_next, status, distance_error = cv.calcOpticalFlowPyrLK(im_mat_0,
                                                                           im_mat_1,
                                                                           features_to_track,
                                                                           reference_features,
                                                                           **lk_parameters)

        return features_in_next, status, distance_error

    @staticmethod
    def track_features_with_object_detection():
        ...
