import cv2
import numpy as np
from pathlib import Path

# Load YOLOv5 model
model = cv2.dnn.readNetFromONNX("yolov5s.onnx")

# Load class names
classes = []
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load camera
cap = cv2.VideoCapture(0)

# Known physical dimensions of the reference object in centimeters or meters
reference_width_cm = 5  # Example: Width of the reference object in centimeters
reference_width_px = 100  # Example: Width of the reference object in pixels

# Calculate conversion factor
conversion_factor = reference_width_cm / reference_width_px

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get dimensions of the frame
    height, width, _ = frame.shape

    # Create a blob from the frame and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (640, 640), [0, 0, 0], 1, crop=False)
    model.setInput(blob)

    # Forward pass through the network
    outputs = model.forward()

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]} {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Measure object dimensions (width and height in pixels)
                object_width_px = w
                object_height_px = h

                # Convert object dimensions from pixels to centimeters
                object_width_cm = object_width_px * conversion_factor
                object_height_cm = object_height_px * conversion_factor

                print(f"Object Dimensions: Width = {object_width_cm:.2f} cm, Height = {object_height_cm:.2f} cm")

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
