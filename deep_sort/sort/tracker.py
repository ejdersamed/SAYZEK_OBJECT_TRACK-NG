import numpy as np
from deep_sort.utils.matching import match_detections
from deep_sort.sort.kalman_filter import CustomKalmanFilter

class Track:
    count = 0

    def __init__(self, bbox, feature):
        self.kalman_filter = CustomKalmanFilter(bbox)
        self.track_id = Track.count
        Track.count += 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.feature = feature

    def predict(self):
        self.kalman_filter.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, bbox, feature):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.feature = feature
        self.kalman_filter.update(bbox)

    def get_state(self):
        return self.kalman_filter.get_state()

class Tracker:
    def __init__(self, max_age=20, min_hits=3):
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        if len(detections) == 0:
            matches, unmatched_dets, unmatched_tracks = [], [], list(range(len(self.tracks)))
        else:
            current_detections = [(det[0], det[1]) for det in detections]
            current_tracks = [(trk.get_state(), trk.feature) for trk in self.tracks]
            matches, unmatched_dets, unmatched_tracks = match_detections(current_detections, current_tracks)

        for det_idx, trk_idx in matches:
            det_bbox, det_feat = detections[det_idx]
            self.tracks[trk_idx].update(det_bbox, det_feat)

        for det_idx in unmatched_dets:
            det_bbox, det_feat = detections[det_idx]
            self.tracks.append(Track(det_bbox, det_feat))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def get_active_tracks(self):
        active_tracks = []
        for track in self.tracks:
            if (track.hits >= self.min_hits) and (track.time_since_update <= 1):
                active_tracks.append(track)
        return active_tracks
