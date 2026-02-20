import cv2
import onnxruntime as ort
import numpy as np
import os, time, datetime, requests, threading
import psutil

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH     = "/home/arm/road_detection/best.onnx"
OUTPUT_DIR     = "/home/arm/road_detection/output"
GDRIVE_FOLDER  = "1ioQ-KAa2ym53LKowGlqi44vMibXru8WT"
SERVICE_ACCT   = "/home/arm/road_detection/service_account.json"
TG_TOKEN       = "8296089902:AAHFV6-aCp1z2j_go1gGGJRs4GVPt-G4d90"
TG_CHAT_ID     = "7068162208"
CONF_THRESH    = 0.25
IOU_THRESH     = 0.45
INPUT_SIZE     = (640, 640)
GDRIVE_INTERVAL = 300
CAM_INDEX      = 0
NUM_CLASSES    = 4

CLASS_NAMES = ["crack", "open_manhole", "pothole", "rutting"]
COLORS = [(0,0,255), (0,165,255), (0,255,0), (255,0,0)]
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── GPS ──────────────────────────────────────────────────────────────────────
def get_location():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5)
        loc = r.json().get("loc", "0,0").split(",")
        return float(loc[0]), float(loc[1])
    except Exception:
        return None, None

def make_location_txt(lat, lon, timestamp):
    path = os.path.join(OUTPUT_DIR, "location_" + timestamp + ".txt")
    if lat and lon:
        gmaps = "https://maps.google.com/?q=" + str(lat) + "," + str(lon)
        content = "Timestamp : " + timestamp + "\nLatitude  : " + str(lat) + "\nLongitude : " + str(lon) + "\nGoogle Map: " + gmaps + "\n"
    else:
        content = "Timestamp : " + timestamp + "\nLocation  : Unavailable\n"
    with open(path, "w") as f:
        f.write(content)
    return path

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
def send_telegram(frame_path, loc_path, lat, lon):
    try:
        url = "https://api.telegram.org/bot" + TG_TOKEN
        caption = "Road Anomaly Detected!\nLat: " + str(lat) + " Lon: " + str(lon)
        if lat and lon:
            caption += "\nhttps://maps.google.com/?q=" + str(lat) + "," + str(lon)
        with open(frame_path, "rb") as img:
            requests.post(url + "/sendPhoto",
                data={"chat_id": TG_CHAT_ID, "caption": caption},
                files={"photo": img}, timeout=15)
        with open(loc_path, "rb") as doc:
            requests.post(url + "/sendDocument",
                data={"chat_id": TG_CHAT_ID},
                files={"document": doc}, timeout=15)
        print("[Telegram] Sent successfully")
    except Exception as e:
        print("[Telegram] Failed:", e)

# ─── GOOGLE DRIVE ─────────────────────────────────────────────────────────────
def upload_to_gdrive(file_path):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCT,
            scopes=["https://www.googleapis.com/auth/drive"])
        service = build("drive", "v3", credentials=creds)
        fname = os.path.basename(file_path)
        ext = fname.split(".")[-1].lower()
        mime = "video/mp4" if ext == "mp4" else "text/plain" if ext == "txt" else "image/jpeg"
        meta = {"name": fname, "parents": [GDRIVE_FOLDER]}
        media = MediaFileUpload(file_path, mimetype=mime, resumable=True)
        service.files().create(body=meta, media_body=media, fields="id").execute()
        print("[Drive] Uploaded:", fname)
    except Exception as e:
        print("[Drive] Failed:", e)

# ─── ONNX INFERENCE ───────────────────────────────────────────────────────────
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(frame):
    img = cv2.resize(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[np.newaxis, :]

def postprocess(outputs, orig_shape):
    # Output shape: (1, 8, 8400) → 4 box coords + 4 classes
    pred = outputs[0][0]          # shape: (8, 8400)
    pred = pred.T                 # shape: (8400, 8)

    boxes_xywh = pred[:, :4]     # first 4 = box
    class_scores = pred[:, 4:]   # last 4 = class scores

    cls_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_scores)), cls_ids]

    mask = confs > CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # Convert xywh to xyxy
    h, w = orig_shape[:2]
    sx = w / INPUT_SIZE[0]
    sy = h / INPUT_SIZE[1]

    results = []
    boxes_for_nms = []
    for box in boxes_xywh:
        cx, cy, bw, bh = box
        x1 = int((cx - bw/2) * sx)
        y1 = int((cy - bh/2) * sy)
        x2 = int((cx + bw/2) * sx)
        y2 = int((cy + bh/2) * sy)
        boxes_for_nms.append([x1, y1, x2-x1, y2-y1])

    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confs.tolist(), CONF_THRESH, IOU_THRESH)
    if len(indices) == 0:
        return []

    for i in indices.flatten():
        x1, y1, bw, bh = boxes_for_nms[i]
        results.append((x1, y1, x1+bw, y1+bh, float(confs[i]), int(cls_ids[i])))

    return results

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUTPUT_DIR, "video_" + ts + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(video_path, fourcc, 20, (1280, 720))

    last_upload_time = time.time()
    last_detect_time = 0
    print("[INFO] Detection started. Press Ctrl+C to stop.")
    frame_start = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed, retrying...")
                time.sleep(0.1)
                continue

           # Inference with timing
            infer_start = time.time()
            inp = preprocess(frame)
            outputs = session.run(None, {input_name: inp})
            detections = postprocess(outputs, frame.shape)
            infer_time = (time.time() - infer_start) * 1000  # ms

            # FPS calculation
            frame_end = time.time()
            fps = 1.0 / (frame_end - frame_start + 0.0001)
            frame_start = frame_end

            # CPU usage
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent

            # Print to terminal
            print(f"FPS: {fps:.1f} | Inference: {infer_time:.1f}ms | CPU: {cpu:.1f}% | RAM: {mem:.1f}% | Detections: {len(detections)}", end="\r")

            anomaly_detected = len(detections) > 0

            for (x1, y1, x2, y2, conf, cls) in detections:
                color = COLORS[cls % len(COLORS)]
                label = CLASS_NAMES[cls] + " " + str(round(conf, 2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, now_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            status = "ANOMALY DETECTED" if anomaly_detected else "Monitoring..."
            color = (0, 0, 255) if anomaly_detected else (0, 255, 0)
            cv2.putText(frame, status, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out_vid.write(frame)

            # Send to telegram (max once every 30 seconds)
            if anomaly_detected and (time.time() - last_detect_time > 30):
                lat, lon = get_location()
                det_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_path = os.path.join(OUTPUT_DIR, "detected_" + det_ts + ".jpg")
                loc_path = make_location_txt(lat, lon, det_ts)
                cv2.imwrite(frame_path, frame)
                threading.Thread(
                    target=send_telegram,
                    args=(frame_path, loc_path, lat, lon),
                    daemon=True).start()
                last_detect_time = time.time()
                print("[DETECT] Anomaly! Lat:", lat, "Lon:", lon)

            # Every 5 mins upload video
            if time.time() - last_upload_time >= GDRIVE_INTERVAL:
                out_vid.release()
                old_video = video_path
                threading.Thread(target=upload_to_gdrive, args=(old_video,), daemon=True).start()
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(OUTPUT_DIR, "video_" + ts + ".mp4")
                out_vid = cv2.VideoWriter(video_path, fourcc, 20, (1280, 720))
                last_upload_time = time.time()
                print("[Drive] New video segment:", video_path)

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        out_vid.release()
        cap.release()
        upload_to_gdrive(video_path)

if __name__ == "__main__":
    main()
