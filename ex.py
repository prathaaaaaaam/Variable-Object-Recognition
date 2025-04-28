import cv2
import torch
from ultralytics import YOLO

# Load YOLO model (replace "yolov8n.pt" with your desired model)
model = YOLO("yolov8n.pt")
model.eval()  # Set model to evaluation mode

# Define frame dimensions (optional, adjust if needed)
frame_width = 640
frame_height = 480

# Open the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame (optional, adjust if needed)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Convert frame to BGR (YOLOv8 expects BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Perform inference (YOLOv8 handles pre-processing internally)
    results = model(frame)

    # Process detections
    for detection in results.pandas().xyxy[0]:  # Assuming single image batch
        x1, y1, x2, y2, conf, cls, cls_id = detection.values

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Display class name and confidence (if class names available)
        if cls:  # Check if class name exists
            class_name = cls.split(" ")[0]  # Extract class name from label
            cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
