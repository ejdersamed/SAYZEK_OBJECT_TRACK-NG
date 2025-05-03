import motmetrics as mm

# Dosya yollarını ayarlayın
gt_file = 'gt/gt.txt'
hyp_file = 'results/results.txt'

# MOT formatındaki dosyaları yükle
def load_mot_file(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split(',')
            frame = int(fields[0])
            id = int(fields[1])
            x, y, w, h = map(float, fields[2:6])
            if frame not in data:
                data[frame] = {}
            data[frame][id] = (x, y, w, h)
    return data

# Bounding box IoU hesaplama
def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1 + w1, x2 + w2)
    yy2 = min(y1 + h1, y2 + h2)
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    intersection = w * h
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0

# GT ve tahminleri yükle
gt_data = load_mot_file(gt_file)
hyp_data = load_mot_file(hyp_file)

acc = mm.MOTAccumulator(auto_id=True)

# Kare kare eşleştirme
for frame in sorted(gt_data.keys()):
    gt_ids = list(gt_data[frame].keys())
    hyp_ids = list(hyp_data.get(frame, {}).keys())
    distances = []

    for gt_id in gt_ids:
        row = []
        for hyp_id in hyp_ids:
            iou_score = iou(gt_data[frame][gt_id], hyp_data[frame][hyp_id])
            row.append(1 - iou_score)  # motmetrics distance = 1 - IoU
        distances.append(row)

    acc.update(gt_ids, hyp_ids, distances)

# Metrikleri hesapla
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1', 'id_switches'], name='summary')
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap={
    'num_frames': 'Frames',
    'mota': 'MOTA',
    'idf1': 'IDF1',
    'id_switches': 'ID Switches'
}))
