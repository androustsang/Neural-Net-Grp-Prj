import base64
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

def process_yolo_prediction(model, file_storage):
    """
    Args:
        model: The loaded YOLO model
        file_storage: The 'file' object from request.files['image']
    """
    
    # --- STEP 1: CONVERT FLASK FILE TO OPENCV IMAGE ---
    # We cannot use cv2.imread() because the file isn't on the disk.
    # We read the raw bytes from memory.
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    
    # Decode the bytes into an image array (BGR format)
    # This is the exact equivalent of 'image = cv2.imread(...)'
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- STEP 2: RUN INFERENCE ---
    # Run the model on the image array
    results = model(image)[0]

    # --- STEP 3: SUPERVISION LOGIC (Custom Code) ---
    
    # Convert to Supervision format
    detections = sv.Detections.from_ultralytics(results)

    # Calculate count (custom logic)
    pothole_count = len(detections)

    # Setup Annotators
    # You can customize thickness, text_scale, and colors here
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Create a copy of the image to draw on (scene=...)
    annotated_image = image.copy()

    # Draw Boxes
    annotated_image = box_annotator.annotate(
        scene=annotated_image, 
        detections=detections
    )
    
    # Draw Labels (e.g., "pothole 0.85")
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections
    )

    # --- STEP 4: ENCODE FOR WEB (NumPy -> Base64) ---
    
    # We cannot send a NumPy array directly to a browser. 
    # We must encode it as a JPEG image in memory.
    _, buffer = cv2.imencode('.jpg', annotated_image)
    
    # Convert that buffer to a Base64 string
    img_str = base64.b64encode(buffer).decode("utf-8")

    return {
        "count": pothole_count,
        "image_data": f"data:image/jpeg;base64,{img_str}"
    }