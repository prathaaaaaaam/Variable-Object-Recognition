import cv2
from ultralytics import YOLO

# Load YOLO model (replace "yolov8n.pt" with your desired model)
model = YOLO("yolov8n.pt")
model.eval()  # Set model to evaluation mode

# Open the camera capture (use camera index 0 for the default camera)
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

    # Convert frame to RGB (YOLOv8 expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference (YOLOv8 handles pre-processing internally)
    results = model(frame_rgb)

    # Process detections
    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls_id = obj[:6]  # Extract relevant information
        class_name = model.names[int(cls_id)]

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Display class name and confidence
        cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame with detected objects
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
