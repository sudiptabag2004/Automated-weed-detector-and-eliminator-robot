from picamera2 import Picamera2
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = "/home/pi/Downloads/best.pt"  # Path to your YOLOv8 model
model = YOLO(model_path)

# Initialize the Pi camera using picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Convert the frame from 4 channels (RGBA) to 3 channels (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    # Run YOLOv8 detection on the frame
    results = model.predict(frame_rgb, conf=0.5)  # Adjust confidence threshold as needed
    annotated_frame = results[0].plot()           # Visualize detections on the frame

    # Display the frame with YOLOv8 detections
    cv2.imshow("Weed Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
picam2.stop()
cv2.destroyAllWindows()
