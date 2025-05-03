import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):

    #İki kutu arasında IoU (Intersection over Union) hesabı.

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def match_detections(dets, tracks, alpha=0.7):
    """
    Detections ile mevcut track'leri eşleştirir.

    Args:
        dets: [(bbox, feature)]
        tracks: [(bbox, feature)]
        alpha: appearance (cosine) ağırlığı. (0.0 - 1.0)

    Returns:
        matches: [(detection_index, track_index)]
        unmatched_dets: [detection_index]
        unmatched_tracks: [track_index]
    """
    cost_matrix = np.zeros((len(dets), len(tracks)), dtype=np.float32)

    for i, (det_bbox, det_feat) in enumerate(dets):
        for j, (trk_bbox, trk_feat) in enumerate(tracks):
            cosine_dist = cosine(det_feat, trk_feat)  # Görsel benzerlik
            iou_score = iou(det_bbox, trk_bbox)        # Kutu örtüşme oranı

            # Cost hesapla: düşük cost daha iyidir
            cost = alpha * cosine_dist + (1 - alpha) * (1 - iou_score)
            cost_matrix[i, j] = cost

    # Hungarian algoritması ile eşleştirme yap
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_dets = list(range(len(dets)))
    unmatched_tracks = list(range(len(tracks)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0.8:  # Eşik değer (cost çok yüksekse eşleşme yapma)
            continue
        matches.append((r, c))
        unmatched_dets.remove(r)
        unmatched_tracks.remove(c)

    return matches, unmatched_dets, unmatched_tracks





"""
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
                cost = 1 - (0.3 * iou_score + 0.7 * sim_score)
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
"""
