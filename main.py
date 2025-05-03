#RESNET FPS ÖLÜÇ EKLENMİŞ
import cv2
import os
import torch
import time
from deep_sort.deep.feature_extractor import ResNet18FeatureExtractor
from deep_sort.sort.tracker import Tracker
from yolo.yolo_infer import YOLODetector
import matplotlib.pyplot as plt

# 📂 Video dosya yolu
video_path = os.path.join('videos', 'İMU_video.mp4')
cap = cv2.VideoCapture(video_path)

# 🎨 15 farklı renk oluştur
def get_distinct_colors(n):
    cmap = plt.get_cmap("tab20")
    return [tuple(int(255 * c) for c in cmap(i % 20)[:3]) for i in range(n)]

colors = get_distinct_colors(15)

# ⚙️ Cihaz ve bileşenler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = ResNet18FeatureExtractor(device=device)
yolo = YOLODetector()
tracker = Tracker(max_age=800, min_hits=3)  # ⬅️ ID kaybı engellendi

# ⏱ FPS ölçüm
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

    cv2.imshow("DeepSORT with ResNet18", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




#RESNET İÇİN ,FPS Yok
"""
import cv2
import os
import torch
from deep_sort.deep.feature_extractor import ResNet18FeatureExtractor
from deep_sort.sort.tracker import Tracker
from yolo.yolo_infer import YOLODetector  # YOLOv5/8 sınıfın varsa
import matplotlib.pyplot as plt

# 📂 Video dosya yolu
video_path = os.path.join('videos', 'İMU_video.mp4')
cap = cv2.VideoCapture(video_path)

# 🎨 15 farklı renk oluştur
def get_distinct_colors(n):
    cmap = plt.get_cmap("tab20")
    return [tuple(int(255 * c) for c in cmap(i % 20)[:3]) for i in range(n)]

colors = get_distinct_colors(15)

# ⚙️ Bileşenleri başlat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = ResNet18FeatureExtractor(device=device)
yolo = YOLODetector() ffffffffff

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ YOLO ile tespit
    detections_raw = yolo.detect(frame)

    # 2️⃣ Görüntüden crop al ve embedding çıkar
    detections = []
    for det in detections_raw:
        x1, y1, x2, y2, conf = det
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(crop_rgb)
        feature = extractor.extract(pil_img).cpu().numpy()
        detections.append(((x1, y1, x2, y2), feature))

    # 3️⃣ Takibi güncelle
    tracker.update(detections)

    # 4️⃣ Aktif track'leri çiz
    for track in tracker.get_active_tracks():
        x1, y1, x2, y2 = track.get_state()
        track_id = track.track_id
        color = colors[track_id % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5️⃣ Ekrana göster
    cv2.imshow("DeepSORT + ResNet18", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""