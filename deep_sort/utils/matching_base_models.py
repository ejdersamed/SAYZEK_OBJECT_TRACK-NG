import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return 1 - cosine(vec1, vec2)

def build_cost_matrix(detections, tracks, use_embedding=True):
    cost_matrix = []
    for det_bbox, det_feat in detections:
        row = []
        for track_bbox, track_feat in tracks:
            iou_score = iou(det_bbox, track_bbox)
            if use_embedding:
                sim_score = cosine_similarity(det_feat, track_feat)
                cost = 1 - (0.7 * iou_score + 0.3 * sim_score)
            else:
                cost = 1 - iou_score
            row.append(cost)
        cost_matrix.append(row)
    return np.array(cost_matrix)

def match_detections(detections, tracks, iou_thresh=0.3):
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(detections))), list(range(len(tracks)))

    cost_matrix = build_cost_matrix(detections, tracks)
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches, unmatched_dets, unmatched_tracks = [], [], []

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r][c] > (1 - iou_thresh):
            unmatched_dets.append(r)
            unmatched_tracks.append(c)
        else:
            matches.append((r, c))

    unmatched_dets += [r for r in range(len(detections)) if r not in row_idx]
    unmatched_tracks += [c for c in range(len(tracks)) if c not in col_idx]

    return matches, unmatched_dets, unmatched_tracks