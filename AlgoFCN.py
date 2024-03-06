# Function to detect and compute features using your custom extractor
from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
from lightglue.utils import load_image_crop, rbd, load_image, read_image
from lightglue import viz2d
import torch
import os
import glob
import copy
import numpy as np
import cv2
import time
import FNCs
import colorsys
import sys

class FeatureTracker:
    def __init__(self, ftExtractor = 'aliked', mx_keypoints = 1024, desired_device = 3, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        # self.extractor = ALIKED(max_num_keypoints=1024).eval().to(self.device)
        # self.matcher = LightGlue(features="aliked").eval().to(self.device)
        self.extractor, self.matcher = self.set_extractor_matcher(ftExtractor, mx_keypoints)
        self.frame_keypoints = {}

        self.tracks = {}
        self.keyp_trackId_dic = {}
        self.global_keyp_trackId_dic = {}
        self.track_id = 0
        self.prev_keypoints = None
        self.prev_feats = None
        self.kpMvmt = 0
        self.last_cropCoords = (0, 0, 0, 0)
        self.trackColors = self.generate_distinct_colors(40)

    def detect_features(self, frame):
        feats = self.extractor.extract(frame.to(self.device))
        feats_rbd = rbd(copy.deepcopy(feats))
        keypoints = feats_rbd["keypoints"]
        return keypoints, feats

    def process_frame__old(self, frameData, crnt_frm_idx, cropCoords):
        cropCoords =(cropCoords[0], cropCoords[1], cropCoords[2], cropCoords[3])

        # # Check if cropCoords has changed since the last frame processed
        # if cropCoords != self.last_cropCoords:
        #     self.tracks = {}  # Reset tracks
        #     self.track_id = 0  # Reset track_id as well, assuming you're starting fresh with tracking
        #     self.last_cropCoords = cropCoords  # Update last_cropCoords to the current one

        frame = FNCs.load_data_image_crop(frameData, crop=cropCoords)
        keypoints, feats = self.detect_features(frame)
        # Convert keypoints to integers
        keypoints = keypoints.round().to(torch.int)

        if crnt_frm_idx == 0:
            # Initialize tracks with the first frame's keypoints
            for kp in keypoints:
                kp = kp.tolist()
                self.tracks[self.track_id] = [(crnt_frm_idx, tuple(kp))]
                # self.keyp_trackId_dic[(crnt_frm_idx, tuple(kp))] = self.track_id
                self.global_keyp_trackId_dic[tuple(kp)] = self.track_id
                self.track_id += 1

        else:
            # Match features with the previous frame
            matches01 = self.matcher({"image0": self.prev_feats, "image1": feats})
            matches01 = rbd(matches01)
            matches = matches01["matches"]
            m_kpts0, m_kpts1 = self.prev_keypoints[matches[..., 0]], keypoints[matches[..., 1]]

            differences = m_kpts0 - m_kpts1
            differences_np = differences.cpu().numpy()  # Minimize transfers

            # Compute movement statistics
            avg_change = differences_np.mean(axis=0)
            self.kpMvmt = avg_change[1]

            m_kpts0, m_kpts1 = m_kpts0.tolist(), m_kpts1.tolist()
            # Update tracks with matched features
            _new_keyp_trackId_dic = {}
            for match_idx, (prev_kp_idx, current_kp_idx) in enumerate(matches):
                prev_kp = self.prev_keypoints[prev_kp_idx].tolist()  # Convert to tuple to use as dictionary key
                current_kp = keypoints[current_kp_idx].tolist() # Matched keypoint in the current frame

                # _track_id_idx = self.keyp_trackId_dic.get((crnt_frm_idx-1, tuple(prev_kp)), None)
                _track_id_idx = self.global_keyp_trackId_dic.get(tuple(prev_kp), None)

                if (_track_id_idx is not None):

                    self.tracks[_track_id_idx].append((crnt_frm_idx, tuple(current_kp)))
                    # FNCs.replace_key(self.keyp_trackId_dic, (crnt_frm_idx-1, tuple(prev_kp)),
                    #                             (crnt_frm_idx, tuple(current_kp)))
                    _new_keyp_trackId_dic[tuple(current_kp)] = _track_id_idx                

                else:
                    print(self.track_id, crnt_frm_idx-1, tuple(prev_kp), "track id not found")
            
            matched_indices = set(matches[..., 1].tolist())
            notMatchKPs = [kp.tolist() for idx, kp in enumerate(keypoints) if idx not in matched_indices]

            for nmkp in notMatchKPs:
                self.tracks[self.track_id] = [(crnt_frm_idx, tuple(nmkp))]
                # self.keyp_trackId_dic[(crnt_frm_idx, tuple(nmkp))] = self.track_id
                _new_keyp_trackId_dic[tuple(nmkp)] = self.track_id
                self.track_id += 1
            
            self.global_keyp_trackId_dic = _new_keyp_trackId_dic
            
        self.prev_keypoints, self.prev_feats = keypoints, feats

        return self.tracks, self.kpMvmt

    def generate_distinct_colors(self, n):
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.9
            lightness = 0.6
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            # Convert from 0-1 range to 0-255 range for each color component
            rgb_255 = tuple(int(c * 255) for c in rgb)
            colors.append(rgb_255)
        return colors

    def process_frame(self, frameData, crnt_frm_idx, cropCoords, isOnline):
        if cropCoords == None:
            cropCoords = (0, 0, 100000, 100000)
        cropCoords = tuple(cropCoords)
        if isOnline:
            frame = FNCs.load_data_image_crop(frameData, crop=cropCoords)
        else:
            frame, image_cv2 = load_image_crop(frameData, crop=cropCoords)

        keypoints, feats = self.detect_features(frame)
        # Convert keypoints to integers
        keypoints = keypoints.round().to(torch.int)
        _frmTracks = []
        self.frame_keypoints[crnt_frm_idx] = []

        if crnt_frm_idx == 0:
            # Initialize tracks with the first frame's keypoint
            for kp in keypoints:
                kp = kp.tolist()
                self.tracks[self.track_id] = [(crnt_frm_idx, tuple(kp))]
                # self.keyp_trackId_dic[(crnt_frm_idx, tuple(kp))] = self.track_id
                self.global_keyp_trackId_dic[tuple(kp)] = self.track_id
                _frmTracks.append(self.track_id)
                self.track_id += 1
                self.frame_keypoints[crnt_frm_idx].append(kp)
        else:
            # Match features with the previous frame
            matches01 = self.matcher({"image0": self.prev_feats, "image1": feats})
            matches01 = rbd(matches01)
            matches = matches01["matches"]
            m_kpts0, m_kpts1 = self.prev_keypoints[matches[..., 0]], keypoints[matches[..., 1]]
            differences_np = (m_kpts0 - m_kpts1).cpu().numpy()  # Minimize transfers

            # Compute movement statistics
            avg_change = differences_np.mean(axis=0)
            self.kpMvmt = avg_change[1]

            # m_kpts0, m_kpts1 = m_kpts0.tolist(), m_kpts1.tolist()
            # Update tracks with matched features
            _new_keyp_trackId_dic = {}

            for match_idx, (prev_kp_idx, current_kp_idx) in enumerate(matches):

                prev_kp = self.prev_keypoints[prev_kp_idx].tolist()  # Convert to tuple to use as dictionary key
                current_kp = keypoints[current_kp_idx].tolist() # Matched keypoint in the current frame

                # _track_id_idx = self.keyp_trackId_dic.get((crnt_frm_idx-1, tuple(prev_kp)), None)

                _track_id_idx = self.global_keyp_trackId_dic.get(tuple(prev_kp), None)
                if (_track_id_idx is not None):
                    self.tracks[_track_id_idx].append((crnt_frm_idx, tuple(current_kp)))
                    # FNCs.replace_key(self.keyp_trackId_dic, (crnt_frm_idx-1, tuple(prev_kp)),
                    #                             (crnt_frm_idx, tuple(current_kp)))
                    _new_keyp_trackId_dic[tuple(current_kp)] = _track_id_idx                
                    _frmTracks.append(_track_id_idx)
                    self.frame_keypoints[crnt_frm_idx].append(current_kp)
                else:
                    print(self.track_id, crnt_frm_idx-1, tuple(prev_kp), "track id not found")

            matched_indices1 = set(matches[..., 1].tolist())
            notMatchKPs1 = [kp.tolist() for idx, kp in enumerate(keypoints) if idx not in matched_indices1]

            for nmkp in notMatchKPs1:

                self.tracks[self.track_id] = [(crnt_frm_idx, tuple(nmkp))]
                # self.keyp_trackId_dic[(crnt_frm_idx, tuple(nmkp))] = self.track_id
                _new_keyp_trackId_dic[tuple(nmkp)] = self.track_id
                self.track_id += 1
            
            matched_indices0 = set(matches[..., 0].tolist())
            notMatchKPs0 = [kp.tolist() for idx, kp in enumerate(keypoints) if idx not in matched_indices0]

            for nmkp in notMatchKPs0:
                # _new_keyp_trackId_dic[tuple(nmkp)] = self.track_id
                _new_keyp_trackId_dic.pop(self.track_id, None)

            self.global_keyp_trackId_dic = _new_keyp_trackId_dic
        
        self.prev_keypoints, self.prev_feats = keypoints, feats
        return self.tracks, self.kpMvmt, image_cv2, _frmTracks

    def update_cvFrame(self, _cpTracks, _frameData, _frmTrackIds, desRec, _isOnline = False):
        
        _cvFrame = _frameData
        if _isOnline:
            _cvFrame = cv2.imdecode(np.frombuffer(_frameData, np.uint8), cv2.IMREAD_COLOR)
        # First, filter out tracks with less than 3 keypoints to avoid modifying the dictionary during iteration
        
        for i, trackId in enumerate(_frmTrackIds):
            frm_keypoints = _cpTracks[trackId]
            color = self.trackColors[trackId % 40]        
            # for idx, (frm, keypoint) in enumerate(frm_keypoints):
            for idx in range(len(frm_keypoints)):
                if idx == len(frm_keypoints)-1:
                    cv2.circle(_cvFrame, frm_keypoints[idx][1], 5, color, -1)
                else:
                    cv2.line(_cvFrame, frm_keypoints[idx][1], frm_keypoints[idx+1][1], color, 2)
            # print(idx)
            # for idp, _ , _ in enumerate(cp)
        return _cvFrame
        _cpTracks = {key: value for key, value in _cpTracks.items() if len(value) >= 3}
        # Now, process the tracks for drawing
        for _track_id, _keypoints in _cpTracks.items():
            # Pre-compute the existence of the current frame in keypoints to avoid repeated checks
            current_frame_exists = any(f == _crnt_frm_idx for f, _ in _keypoints)

            # If the current frame doesn't exist in the keypoints, skip further processing for this track
            if not current_frame_exists:
                continue

            # Draw each track on the frame
            for f, point in _keypoints:
                if f == _crnt_frm_idx:  # Check if the keypoint belongs to the current frame
                    cv2.circle(_cvFrame, point, 5, (0, 255, 0), -1)  # Draw the keypoint

            # Draw lines between keypoints of the same track
            for kp_idx in range(1, len(_keypoints)):
                f, point = _keypoints[kp_idx]
                if f <= _crnt_frm_idx:
                    start_point = _keypoints[kp_idx-1][1]
                    end_point = point
                    cv2.line(_cvFrame, start_point, end_point, (255, 0, 0), 2)

        return _cvFrame

    def set_extractor_matcher(self, ftExtractor, mx_keypoints):
        extractor, matcher = None, None
        if ftExtractor == 'aliked':
            extractor = ALIKED(max_num_keypoints=mx_keypoints).eval().to(self.device)
            matcher = LightGlue(features=ftExtractor).eval().to(self.device)

        elif ftExtractor == 'superpoint':
            extractor = SuperPoint(max_num_keypoints=mx_keypoints).eval().to(self.device)
            matcher = LightGlue(features=ftExtractor).eval().to(self.device)

        elif ftExtractor == 'disk':
            extractor = DISK(max_num_keypoints=mx_keypoints).eval().to(self.device)
            matcher = LightGlue(features=ftExtractor).eval().to(self.device)

        elif ftExtractor == 'sift':
            extractor = SIFT(max_num_keypoints=mx_keypoints).eval().to(self.device)
            matcher = LightGlue(features=ftExtractor).eval().to(self.device)

        else:
            raise RuntimeError("Invalid feature extractor. Please use 'aliked' or 'superpoint'.")
        
        return extractor, matcher

#*******************
######################
# Additional methods (like replace_key, frame_exists, etc.) should be defined here as needed.

# Example usage:
# tracker = FeatureTracker()
# processed_frame = tracker.process_frame("path/to/frame.jpg", 0)
    

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

# extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
# matcher = LightGlue(features="superpoint").eval().to(device)

# def detect_features(frame, extractor):
#     feats = extractor.extract(frame.to(device))
#     # Remove batch dimension if necessary
#     feats_rbd = rbd(copy.deepcopy(feats))
#     keypoints = feats_rbd["keypoints"].cpu()
#     return keypoints, feats

# def process_frame(framePath, crnt_frm_idx):
#     frame = load_image_crop(framePath, crop=(0, 0, 1024, 480))
#     keypoints, feats = detect_features(frame, extractor)

#     if crnt_frm_idx == 0:
#         # Initialize tracks with the first frame's keypoints
#         for kp in keypoints:
#             kp = kp.tolist()
#             process_frame.tracks[process_frame.track_id] = [(crnt_frm_idx, tuple(kp))]
#             process_frame.keyp_trackId_dic[(crnt_frm_idx, tuple(kp))] = process_frame.track_id
#             process_frame.track_id += 1

#     else:
#         # Match features with the previous frame
#         matches01 = matcher({"image0": process_frame.prev_feats, "image1": feats})
#         matches01 = rbd(matches01)
#         matches = matches01["matches"]
#         m_kpts0, m_kpts1 = process_frame.prev_keypoints[matches[..., 0]], keypoints[matches[..., 1]]

#         differences = m_kpts0 - m_kpts1
#         differences_np = differences.cpu().numpy()

#         average_change_x = np.mean(differences_np[:, 0])
#         average_change_y = np.mean(differences_np[:, 1])

#         variance_x = np.var(differences_np[:, 0])
#         variance_y = np.var(differences_np[:, 1])

#         # Calculate the standard deviation for changes in x and y
#         std_deviation_x = np.sqrt(variance_x)
#         std_deviation_y = np.sqrt(variance_y)

#         process_frame.kpMvmt[crnt_frm_idx] = (average_change_x, average_change_y, variance_x, variance_y, std_deviation_x, std_deviation_y)

#         m_kpts0, m_kpts1 = m_kpts0.tolist(), m_kpts1.tolist()
#         # Update tracks with matched features
#         for match_idx, (prev_kp_idx, current_kp_idx) in enumerate(matches):
#             prev_kp = process_frame.prev_keypoints[prev_kp_idx].tolist()  # Convert to tuple to use as dictionary key
#             current_kp = keypoints[current_kp_idx].tolist() # Matched keypoint in the current frame

#             _track_id_idx = process_frame.keyp_trackId_dic.get((crnt_frm_idx-1, tuple(prev_kp)), None)
#             if (_track_id_idx is not None) and (_track_id_idx != process_frame.global_track_id):

#                 process_frame.tracks[_track_id_idx].append((crnt_frm_idx, tuple(current_kp)))
#                 FNCs.replace_key(process_frame.keyp_trackId_dic, (crnt_frm_idx-1, tuple(prev_kp)),
#                                               (crnt_frm_idx, tuple(current_kp)))                
#                 process_frame.global_track_id = _track_id_idx

#             else:
#                 print(process_frame.track_id, crnt_frm_idx-1, tuple(prev_kp), "track id not found")
        
#         matched_indices = set(matches[..., 1].tolist())
#         notMatchKPs = [kp.tolist() for idx, kp in enumerate(keypoints) if idx not in matched_indices]

#         for nmkp in notMatchKPs:
#             process_frame.tracks[process_frame.track_id] = [(crnt_frm_idx, tuple(nmkp))]
#             process_frame.keyp_trackId_dic[(crnt_frm_idx, tuple(nmkp))] = process_frame.track_id
#             process_frame.track_id += 1
            
#     process_frame.prev_keypoints, process_frame.prev_feats = keypoints, feats

#     cvFrame = cv2.imread(framePath)
#     # cvFrame = bytes_to_cvFrame(my_frameData(framePath))
#     height, width = cvFrame[0].shape
#     frame_size = (width, height)
#     cpTracks = copy.deepcopy(process_frame.tracks)

#     for key, value in list(cpTracks.items()):
#         if len(value) < 3:
#             del cpTracks[key]

#     for tkId, tkVal in cpTracks.items():
#         # Convert each tuple of floats to a tuple of integers
#         tkVal_int = [(frame, tuple(map(int, coordinates))) for frame, coordinates in tkVal]
#         cpTracks[tkId] = tkVal_int

    
#     for _track_id, _keypoints in cpTracks.items(): # copy_tracks
#         # Draw each track on the frame
#         for f, point in _keypoints:
#             if f == crnt_frm_idx:  # Check if the keypoint belongs to the current frame
#                 cv2.circle(cvFrame, point, 5, (0, 255, 0), -1)  # Draw the keypoint
#                 # Optionally, draw the track ID next to the keypoint
#                 # cv2.putText(frame, str(_track_id), (point[0] + 10, point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

#         # # Draw lines between keypoints of the same track
#         for kp_idx, (f, point) in enumerate(_keypoints):
#             exists = FNCs.frame_exists(_keypoints, crnt_frm_idx)
#             if f <= crnt_frm_idx and kp_idx > 0 and exists:  # Check if the keypoint belongs to the current frame
#                 start_point = _keypoints[kp_idx-1][1]
#                 end_point = _keypoints[kp_idx][1]
#                 cv2.line(cvFrame, start_point, end_point, (255, 0, 0), 2)

#     return cvFrame
