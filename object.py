import cv2
from ultralytics import YOLO

# Constants
CONF_THRESH = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Allowed labels (sets for O(1) lookups)
OBJECT_LABELS = set([
    'cell phone', 'microphone', 'remote', 'laptop',
    'keyboard', 'mouse', 'camera', 'glasses', 'tv', 'bottle'
])
ANIMAL_LABELS = set(['cat', 'dog', 'bird', 'horse', 'cow', 'sheep'])
ALLOWED_LABELS = OBJECT_LABELS.union(ANIMAL_LABELS)

# Load model
model = YOLO("yolov8n.pt")
model_classes_lower = {id: name.lower() for id, name in model.names.items()}  # Pre-lowercase

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference (disable logging)
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model_classes_lower[cls_id]  # Use pre-lowercased name
            conf = float(box.conf[0])
            if conf < CONF_THRESH or label not in ALLOWED_LABELS:
                continue

            # Assign category/color
            if label in OBJECT_LABELS:
                category, color = "OBJECT", (0, 255, 255)  # Yellow
            else:  # Must be an animal
                category, color = "ANIMAL", (255, 0, 0)  # Blue

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Text with background
            text = f'{category} ({label})'
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Filtered Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()