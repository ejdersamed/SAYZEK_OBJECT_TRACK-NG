
#FPS EKlenmiş Hali
import cv2
import os
import torch
import time
from deep_sort.deep.feature_extractor2_base_models import MobileNetFeatureExtractor  # 👈 Mobilenet_v2 class
from deep_sort.sort.tracker_base_models import Tracker
from yolo.yolo_infer import YOLODetector
import matplotlib.pyplot as plt

# 📂 Video dosya yolu
video_path = os.path.join('videos', 'İMU_video.mp4')
cap = cv2.VideoCapture(video_path)

# 🎨 Renk paleti (15 kişi için farklı renkler)
def get_distinct_colors(n):
    cmap = plt.get_cmap("tab20")
    return [tuple(int(255 * c) for c in cmap(i % 20)[:3]) for i in range(n)]

colors = get_distinct_colors(15)

# ⚙️ Cihaz ve bileşenler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = MobileNetFeatureExtractor(device=device)  # 👈 Mobilenet
yolo = YOLODetector()
tracker = Tracker(max_age=220, min_hits=3)  # 👈 ID switch çözümü

# ⏱ FPS ölçüm başlat
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = yolo.detect(frame)

    detections = []
    for det in results:
        x1, y1, x2, y2, conf = det
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(crop_rgb)
        feature = extractor.extract(pil_img).cpu().numpy()
        detections.append(((x1, y1, x2, y2), feature))

    tracker.update(detections)

    for track in tracker.get_active_tracks():
        x1, y1, x2, y2 = track.get_state()
        track_id = track.track_id
        color = colors[track_id % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 🕒 FPS hesapla
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("DeepSORT + MobileNet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Mobilenet_v2 İÇİN
"""
import cv2
import os
import matplotlib.pyplot as plt

from yolo.yolo_infer import YOLODetector
from deep_sort.deep.feature_extractor import FeatureExtractor
from deep_sort.sort.tracker import Tracker

# 🎨 Ayırt edilebilir renk üret
def get_distinct_colors(n):
    cmap = plt.get_cmap("tab20")
    return [tuple(int(255 * c) for c in cmap(i % 20)[:3]) for i in range(n)]

colors = get_distinct_colors(15)

# 📂 Video dosyasını aç
video_path = os.path.join("videos", "İMU_video.mp4") 
cap = cv2.VideoCapture(video_path)

# 🧠 Algılama, Özellik çıkarımı ve Takip birimlerini başlat
yolo = YOLODetector()
extractor = FeatureExtractor()
tracker = Tracker(max_age=220,min_hits=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO ile tespit
    detections_raw = yolo.detect(frame)

    # 2. Crop + embedding çıkarımı
    detections = []
    for det in detections_raw:
        x1, y1, x2, y2, conf = det
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        feature = extractor.extract(crop)
        detections.append(((x1, y1, x2, y2), feature))

    # 3. Takip işlemleri
    tracker.update(detections)

    # 4. 🎯 Sadece aktif track'leri çiziyoruz!
    for track in tracker.get_active_tracks():
        x1, y1, x2, y2 = track.get_state()
        track_id = track.track_id
        color = colors[track_id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5. Görüntüyü göster
    cv2.imshow("DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Temizlik
cap.release()
cv2.destroyAllWindows()
"""
