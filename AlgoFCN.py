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
import math

class FeatureTracker:
    def __init__(self, ftExtractor = 'aliked', mx_keypoints = 1024, desired_device = 3, device=None):
        self.device = self.setDevice(desired_device)
        self.extractor, self.matcher = self.set_extractor_matcher(ftExtractor, mx_keypoints)
        self.frame_keypoints = {}
        self.tracks = {}
        self.keyp_trackId_dic = {}
        self.global_keyp_trackId_dic = {}
        self.track_id = 0
        self.prev_keypoints = None
        self.prev_feats = None
        self.last_cropCoords = (0, 0, 0, 0)
        self.trackColors = self.generate_manual_colors()
        self.FGT_previous_ids = set()
        self.FGT_all_missing_ids = set()
        self.FGT_frameGrowthingTrack = {}
        self.avg_delta_xy = (0, 0)

    def detect_features(self, frame):
        feats = self.extractor.extract(frame.to(self.device))
        feats_rbd = rbd(copy.deepcopy(feats))
        keypoints = feats_rbd["keypoints"]
        return keypoints, feats

 
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
    def generate_manual_colors(self):
        colors = [
            (0, 255, 255),  # Yellow (BGR format)
            (0, 255, 0),    # Green (BGR format)
            (0, 0, 255),    # Red (BGR format)
            (255, 0, 0),    # Blue (BGR format)
            (255, 0, 255),  # Purple (BGR format)
            (0, 165, 255)   # Orange (BGR format)
        ]
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
        # keypoints = keypoints.round().to(torch.int)
        _frmTracks = []
        kp_lent_ornt_deltaXY_list = []
        self.frame_keypoints[crnt_frm_idx] = []

        if crnt_frm_idx == 0:
            # Initialize tracks with the first frame's keypoint
            for kp in keypoints:
                kp = kp.tolist()
                self.tracks[self.track_id] = [(crnt_frm_idx, tuple(kp))]
                # self.keyp_trackId_dic[(crnt_frm_idx, tuple(kp))] = self.track_id
                self.global_keyp_trackId_dic[tuple(kp)] = self.track_id
                # _frmTracks.append(self.track_id)
                self.track_id += 1
                self.frame_keypoints[crnt_frm_idx].append(kp)
        else:
            # Match features with the previous frame
            matches01 = self.matcher({"image0": self.prev_feats, "image1": feats})
            matches01 = rbd(matches01)
            matches = matches01["matches"]
            m_kpts0, m_kpts1 = self.prev_keypoints[matches[..., 0]], keypoints[matches[..., 1]]
            # differences_np = (m_kpts0 - m_kpts1).cpu().numpy()  # Minimize transfers

            # # Compute movement statistics
            # avg_change = differences_np.mean(axis=0)
            # self.kpMvmt = avg_change[1]

            # m_kpts0, m_kpts1 = m_kpts0.tolist(), m_kpts1.tolist()
            # Update tracks with matched features
            _new_keyp_trackId_dic = {}

            for match_idx, (prev_kp_idx, current_kp_idx) in enumerate(matches):

                prev_kp = self.prev_keypoints[prev_kp_idx].tolist()  # Convert to tuple to use as dictionary key
                current_kp = keypoints[current_kp_idx].tolist() # Matched keypoint in the current frame
                lent, angle, delta_x, delta_y = self.calc_lent_Ornt_deltaXY(prev_kp, current_kp)
                kp_lent_ornt_deltaXY_list.append((lent, angle, delta_x, delta_y))
                _lenar = len(kp_lent_ornt_deltaXY_list)
                self.avg_delta_xy = ((self.avg_delta_xy[0] * _lenar + delta_x)/(_lenar + 1),
                                     (self.avg_delta_xy[1] * _lenar + delta_y)/(_lenar + 1))
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
            notMatchKPs0 = [kp.tolist() for idx, kp in enumerate(self.prev_keypoints) if idx not in matched_indices0]

            for nmkp in notMatchKPs0:
                _track_id_idx = self.global_keyp_trackId_dic.get(tuple(nmkp), None)
                if _track_id_idx != None:
                    self.global_keyp_trackId_dic.pop(tuple(nmkp))
                # self.tracks.pop(_track_id_idx)
                # _frmTracks = [x for x in _frmTracks if x != _track_id_idx]

            self.global_keyp_trackId_dic = _new_keyp_trackId_dic
        
        self.prev_keypoints, self.prev_feats = keypoints, feats

        __frameGrowthingTracks = self.newFrameGrowthingTrack(kp_lent_ornt_deltaXY_list, _frmTracks,
                                                              self.tracks, self.avg_delta_xy)

        return self.tracks, kp_lent_ornt_deltaXY_list, image_cv2, _frmTracks, __frameGrowthingTracks

    def update_cvFrame(self, _cpTracks, _frameData, _frmTrackIds, desRec, _isOnline = False):
        
        _cvFrame = _frameData
        if _isOnline:
            _cvFrame = cv2.imdecode(np.frombuffer(_frameData, np.uint8), cv2.IMREAD_COLOR)
        # First, filter out tracks with less than 3 keypoints to avoid modifying the dictionary during iteration
        
        for i, trackId in enumerate(_frmTrackIds):
            frm_keypoints = _cpTracks[trackId]

            if len(frm_keypoints) > 2:
                color = self.trackColors[trackId % len(self.trackColors)]

                # Iterate until the second-to-last item to avoid IndexError
                for idx in range(len(frm_keypoints) - 1):
                    startPoint = self.toInt(frm_keypoints[idx][1])
                    endPoint = self.toInt(frm_keypoints[idx + 1][1])

                    # Draw line for all but the last point, where a circle will be drawn
                    # cv2.line(_cvFrame, startPoint, endPoint, color, 2)
                    cv2.arrowedLine(_cvFrame, startPoint, endPoint, color, 2, tipLength=0.1)
                # Draw a circle at the last point
                lastPoint = self.toInt(frm_keypoints[-1][1])
                cv2.circle(_cvFrame, lastPoint, 3, color, -1)

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

    def toInt(self, floatTuple):
        return tuple(int(x) for x in floatTuple)
    
    def setDevice(self, desired_device):
        device = None
        if torch.cuda.is_available():
            desired_device = 0  # Set this to your desired device index
            if desired_device < torch.cuda.device_count():
                device = torch.device(f"cuda:{desired_device}")
            else:
                print(f"Warning: Desired device cuda:{desired_device} is not available.")
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        return device
        
    def calc_lent_Ornt_deltaXY(self, prevKP, curnKP):
        x0, y0, x1, y1 = prevKP[0], prevKP[1], curnKP[0], curnKP[1]
        
        delta_x = x1 - x0
        delta_y = y0 - y1  # Inverting Y-axis because origin is top-left
        angle = math.atan2(delta_x, delta_y)  # Swap x and y for angle with vertical axis
        # if angle < 0:
        #     angle += 2 * math.pi

        lent = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        return lent, angle, delta_x, delta_y

    def calc_delta(self, lent, orientation):
        delta_x = lent * math.sin(orientation)
        delta_y = lent * math.cos(orientation)
        return delta_x, delta_y

    def newFrameGrowthingTrack(self, kp_lent_ornt_deltaXY_list, _currFrmTracks, tracks, avg_delta_xy):

        # avg_delta_x = avg_delta_y = 0
        # delta_list = [(delta_x, delta_y) for _, _, delta_x, delta_y in kp_lent_ornt_deltaXY_list]

        # if len(delta_list) != 0:
        #     avg_delta_x = sum(delta[0] for delta in delta_list) / len(delta_list)
        #     avg_delta_y = sum(delta[1] for delta in delta_list) / len(delta_list)

        for key, value in self.FGT_frameGrowthingTrack.items():
            self.FGT_frameGrowthingTrack[key] = [(id, (x + self.avg_delta_xy[0], y + self.avg_delta_xy[1]))
                                                  for id, (x, y) in value]

        for i, tID in enumerate(_currFrmTracks):
            # self.FGT_frameGrowthingTrack[tID] = tracks[tID]
            self.FGT_frameGrowthingTrack[tID] = [(id, (x + 700, 450 - y)) for id, (x, y) in tracks[tID]]

        return self.FGT_frameGrowthingTrack
