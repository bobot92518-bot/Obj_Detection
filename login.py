import cv2
from ultralytics import YOLO

# Load YOLOv8 pre-trained model (nano version for speed)
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' or others

# Start webcam (use 0 for default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Perform detection
    results = model(frame)

    # Annotate frame with detection results
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
