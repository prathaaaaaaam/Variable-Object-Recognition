import cv2
import numpy as np

# Load YOLOv8 model and its configuration
net = cv2.dnn.readNetFromDarknet("yolov8.cfg", "yolov8.weights")

# Load class names
with open("coco.names", "r") as f:
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

    # Convert frame to darknet format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(frame_rgb, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Perform object detection
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            label = f"{classes[int(detections[0, 0, i, 1])]} {confidence:.2f}"
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Measure object dimensions (width and height in pixels)
            object_width_px = end_x - start_x
            object_height_px = end_y - start_y

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