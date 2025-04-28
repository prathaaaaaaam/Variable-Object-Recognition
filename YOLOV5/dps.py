import cv2
import numpy as np
import ultralytics

# Load YOLOv8 model and its configuration
net = cv2.dnn.readNet("yolov8.weights", "yolov8.cfg")
classes = []
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

    # Create a blob from the frame and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Extract information from the network output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
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

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
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
