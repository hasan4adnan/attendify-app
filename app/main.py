"""
Attendify - Face Detection and Recognition Module (Enhanced)

This module performs real-time face detection and recognition using OpenCV and Haar Cascade.
It opens the webcam, detects faces within a defined ellipse region, and compares them with
registered face embeddings. Everything outside the ellipse is masked to black. 
Includes FPS display for performance monitoring.

Requirements:
    - Python 3.x
    - OpenCV (cv2) - install with: pip install opencv-python
    - face_recognition - install with: pip install face-recognition (for recognition)
    - NumPy - install with: pip install numpy

Usage:
    python main.py

Controls:
    - Press 'q' to quit the application

Note:
    Face embeddings should be saved in the 'embeddings/' folder as .npy or .json files.
    If no embeddings are found, the program runs in detection-only mode.
"""

import cv2
import time
import numpy as np
import os
import json
from pathlib import Path

# Use DeepFace with ArcFace (InsightFace) for proper face embeddings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, Exception) as e:
    DEEPFACE_AVAILABLE = False
    DeepFace = None


def load_face_cascade():
    """
    Load the Haar Cascade classifier for face detection.
    
    Returns:
        cv2.CascadeClassifier: The face detection classifier
    """
    # OpenCV includes the Haar cascade file, accessible via cv2.data.haarcascades
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise FileNotFoundError(
            "Haar cascade file not found. Please ensure OpenCV is properly installed."
        )
    
    return face_cascade


def detect_faces(frame, face_cascade):
    """
    Detect faces in a frame and return their coordinates.
    
    Args:
        frame: The video frame (grayscale image)
        face_cascade: The Haar Cascade classifier for face detection
    
    Returns:
        list: List of rectangles (x, y, w, h) representing detected faces
    """
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,      # Scale factor for image pyramid
        minNeighbors=5,       # Minimum neighbors required for detection
        minSize=(30, 30)      # Minimum face size
    )
    
    return faces


def is_point_inside_ellipse(point, center, axes):
    """
    Check if a point is inside an ellipse.
    
    Args:
        point: Tuple (x, y) - the point to check
        center: Tuple (cx, cy) - center of the ellipse
        axes: Tuple (a, b) - half-lengths of the ellipse axes (width, height)
    
    Returns:
        bool: True if point is inside ellipse, False otherwise
    """
    x, y = point
    cx, cy = center
    a, b = axes
    
    # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
    dx = (x - cx) / a
    dy = (y - cy) / b
    return (dx * dx + dy * dy) <= 1.0


def filter_faces_in_ellipse(faces, ellipse_center, ellipse_axes):
    """
    Filter faces to only include those whose center point is inside the ellipse.
    
    Args:
        faces: List of face rectangles (x, y, w, h)
        ellipse_center: Tuple (cx, cy) - center of the ellipse
        ellipse_axes: Tuple (a, b) - half-lengths of the ellipse axes
    
    Returns:
        list: Filtered list of faces inside the ellipse
    """
    filtered_faces = []
    
    for (x, y, w, h) in faces:
        # Calculate the center point of the face rectangle
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Check if the face center is inside the ellipse
        if is_point_inside_ellipse(
            (face_center_x, face_center_y),
            ellipse_center,
            ellipse_axes
        ):
            filtered_faces.append((x, y, w, h))
    
    return filtered_faces


def create_ellipse_mask(frame_shape, center, axes):
    """
    Create a binary mask for the ellipse region.
    
    Args:
        frame_shape: Tuple (height, width) of the frame
        center: Tuple (cx, cy) - center of the ellipse
        axes: Tuple (a, b) - full lengths of the ellipse axes
    
    Returns:
        numpy.ndarray: Binary mask (255 inside ellipse, 0 outside)
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.ellipse(
        mask,
        center,
        axes,
        0,      # Angle
        0,      # Start angle
        360,    # End angle
        255,    # White (inside ellipse)
        -1      # Filled
    )
    return mask


def apply_ellipse_mask(frame, mask):
    """
    Apply the ellipse mask to black out everything outside the ellipse.
    
    Args:
        frame: The video frame
        mask: Binary mask (255 inside ellipse, 0 outside)
    
    Returns:
        numpy.ndarray: Frame with everything outside ellipse blacked out
    """
    # Create a 3-channel mask
    mask_3channel = cv2.merge([mask, mask, mask])
    # Apply mask: keep pixels where mask is 255, set to black where mask is 0
    masked_frame = cv2.bitwise_and(frame, mask_3channel)
    return masked_frame


def draw_detection_ellipse(frame, center, axes, has_face_detected):
    """
    Draw the detection ellipse on the frame. Border stays white normally.
    
    Args:
        frame: The video frame to draw on
        center: Tuple (cx, cy) - center of the ellipse
        axes: Tuple (a, b) - full lengths of the ellipse axes (for cv2.ellipse)
        has_face_detected: Boolean - True if face is detected, False otherwise
    
    Returns:
        numpy.ndarray: Frame with ellipse drawn
    """
    # Ellipse border stays white (as per requirements)
    ellipse_color = (255, 255, 255)  # White (BGR)
    
    # Draw the ellipse with thicker border
    cv2.ellipse(
        frame,
        center,
        axes,  # (a, b) as full axes lengths
        0,     # Angle (0 for vertical ellipse)
        0,     # Start angle
        360,   # End angle
        ellipse_color,
        3      # Thicker border
    )
    
    return frame


def load_embeddings(embeddings_dir="embeddings"):
    """
    Load all face embeddings from the specified directory.
    Supports .npy and .json files.
    
    Args:
        embeddings_dir: String - path to the embeddings directory
    
    Returns:
        list: List of numpy arrays containing face embeddings
    """
    embeddings = []
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        print(f"Embeddings directory '{embeddings_dir}' not found. Running in detection-only mode.")
        return embeddings
    
    # Load .npy files
    npy_files = list(embeddings_path.glob("*.npy"))
    for npy_file in npy_files:
        try:
            embedding = np.load(npy_file)
            # Check embedding dimension (ArcFace produces 512D, old dlib produces 128D)
            if embedding.shape[0] == 128:
                print(f"WARNING: {npy_file.name} appears to be from old dlib system (128D).")
                print(f"  Please re-register using the new ArcFace system (512D embeddings).")
                print(f"  Skipping this embedding.")
                continue
            elif embedding.shape[0] != 512:
                print(f"WARNING: {npy_file.name} has unexpected dimension {embedding.shape[0]}.")
                print(f"  Expected 512D (ArcFace). Skipping.")
                continue
            
            embeddings.append(embedding)
            print(f"Loaded embedding from: {npy_file.name} (512D ArcFace)")
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
    
    # Load .json files
    json_files = list(embeddings_path.glob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Handle different JSON formats
                if isinstance(data, list):
                    embedding = np.array(data)
                elif isinstance(data, dict) and 'embedding' in data:
                    embedding = np.array(data['embedding'])
                else:
                    embedding = np.array(data)
                embeddings.append(embedding)
                print(f"Loaded embedding from: {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if len(embeddings) == 0:
        print(f"No embeddings found in '{embeddings_dir}'. Running in detection-only mode.")
    else:
        print(f"Loaded {len(embeddings)} face embedding(s). Recognition enabled.\n")
    
    return embeddings


def extract_face_embedding(face_image):
    """
    Extract face embedding using DeepFace ArcFace (InsightFace) model.
    This produces 512-dimensional discriminative embeddings suitable for identity recognition.
    
    Args:
        face_image: BGR image containing a face (numpy array)
    
    Returns:
        numpy.ndarray: L2-normalized face embedding vector (512 dimensions), or None if extraction fails
    """
    if not DEEPFACE_AVAILABLE or DeepFace is None:
        return None
    
    try:
        # Convert BGR to RGB (DeepFace expects RGB)
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Extract embedding using ArcFace (InsightFace) - produces 512D embeddings
        # DeepFace handles: face detection, alignment, preprocessing
        embedding_result = DeepFace.represent(
            face_rgb,
            model_name='ArcFace',  # InsightFace/ArcFace - 512D embeddings
            enforce_detection=True,  # Require face detection for quality
            align=True,  # Face alignment for better accuracy
            normalization='base'  # Base normalization
        )
        
        if embedding_result and len(embedding_result) > 0:
            embedding = np.array(embedding_result[0]['embedding'], dtype=np.float32)
            
            # L2 normalize the embedding (critical for proper cosine similarity)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                return None
            
            return embedding
        else:
            return None
            
    except Exception as e:
        # Silently fail (don't spam console during recognition)
        return None


def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two L2-normalized embeddings.
    For normalized vectors, cosine similarity = dot product.
    
    Args:
        embedding1: numpy array - first L2-normalized embedding
        embedding2: numpy array - second L2-normalized embedding
    
    Returns:
        float: Cosine similarity score (-1 to 1, higher is more similar)
               Typical values: Same person: 0.6-0.9, Different people: <0.3
    """
    # Both embeddings should already be L2-normalized
    # For normalized vectors, cosine similarity = dot product
    similarity = np.dot(embedding1, embedding2)
    
    # Clamp to [-1, 1] range (should already be in this range for normalized vectors)
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return float(similarity)


def find_best_match(detected_embedding, stored_embeddings, threshold=0.45):
    """
    Find the best matching embedding from stored embeddings using cosine similarity.
    
    Args:
        detected_embedding: numpy array - L2-normalized embedding of detected face
        stored_embeddings: list of numpy arrays - L2-normalized stored face embeddings
        threshold: float - minimum cosine similarity threshold (default 0.45)
                   Expected: Same person: 0.6-0.9, Different people: <0.3
    
    Returns:
        bool: True if a match is found above threshold, False otherwise
    """
    if len(stored_embeddings) == 0:
        return False
    
    best_similarity = -1.0  # Start with minimum similarity
    
    for stored_embedding in stored_embeddings:
        # Ensure embeddings have same shape
        if detected_embedding.shape != stored_embedding.shape:
            continue
        
        # Normalize stored embedding if not already normalized (for backward compatibility)
        stored_norm = np.linalg.norm(stored_embedding)
        if stored_norm > 0:
            stored_normalized = stored_embedding / stored_norm
        else:
            continue
        
        # Calculate cosine similarity (both should be normalized)
        similarity = cosine_similarity(detected_embedding, stored_normalized)
        best_similarity = max(best_similarity, similarity)
    
    # Debug output
    print(f"Best similarity: {best_similarity:.3f} (threshold: {threshold})")
    
    return best_similarity >= threshold


def extract_face_region(frame, face_rect, padding=20):
    """
    Extract the face region from the frame with padding.
    
    Args:
        frame: The full video frame
        face_rect: Tuple (x, y, w, h) - face rectangle coordinates
        padding: Integer - padding around face in pixels
    
    Returns:
        numpy.ndarray: Cropped face image, or None if extraction fails
    """
    x, y, w, h = face_rect
    frame_height, frame_width = frame.shape[:2]
    
    # Add padding and ensure coordinates are within frame bounds
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame_width, x + w + padding)
    y2 = min(frame_height, y + h + padding)
    
    face_region = frame[y1:y2, x1:x2]
    
    # Ensure we have a valid face region
    if face_region.size > 0 and face_region.shape[0] > 10 and face_region.shape[1] > 10:
        return face_region
    else:
        return None


def display_match_message(frame, ellipse_center, ellipse_axes):
    """
    Display "Match found" message outside the ellipse area.
    
    Args:
        frame: The video frame to draw on
        ellipse_center: Tuple (cx, cy) - center of the ellipse
        ellipse_axes: Tuple (a, b) - full lengths of the ellipse axes
    
    Returns:
        numpy.ndarray: Frame with message drawn
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Position message at the bottom of the frame, outside the ellipse
    # Calculate ellipse bottom position
    ellipse_bottom = ellipse_center[1] + ellipse_axes[1] // 2
    
    # Position message below the ellipse with some margin
    message_y = min(frame_height - 40, ellipse_bottom + 50)
    
    message_text = "Match found"
    text_size = cv2.getTextSize(message_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = (frame_width - text_size[0]) // 2
    
    # Draw background rectangle for better readability
    cv2.rectangle(
        frame,
        (text_x - 15, message_y - text_size[1] - 10),
        (text_x + text_size[0] + 15, message_y + 10),
        (0, 0, 0),  # Black background
        -1
    )
    
    # Draw the message in green
    cv2.putText(
        frame,
        message_text,
        (text_x, message_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),  # Green color
        3
    )
    
    return frame


def main():
    """
    Main function to run the enhanced face detection and recognition application.
    """
    print("Initializing Attendify Face Detection and Recognition Module...")
    print("Press 'q' to quit the application.\n")
    
    # Load stored face embeddings
    stored_embeddings = load_embeddings("embeddings")
    
    # Normalize all stored embeddings (for backward compatibility with old embeddings)
    normalized_stored_embeddings = []
    for emb in stored_embeddings:
        norm = np.linalg.norm(emb)
        if norm > 0:
            normalized_stored_embeddings.append(emb / norm)
        else:
            print(f"Warning: Skipping zero-norm embedding")
    
    stored_embeddings = normalized_stored_embeddings
    recognition_enabled = len(stored_embeddings) > 0 and DEEPFACE_AVAILABLE
    
    if recognition_enabled:
        print("Face recognition enabled (using DeepFace with ArcFace).")
    else:
        print("Running in detection-only mode.")
        if not DEEPFACE_AVAILABLE:
            print("\nTo install TensorFlow for Apple Silicon Mac:")
            print("  1. pip install tensorflow-macos tensorflow-metal")
            print("  2. pip install deepface")
            print("\nNote: tensorflow-metal enables GPU acceleration on Apple Silicon.")
    print()
    
    # Load the face cascade classifier
    try:
        face_cascade = load_face_cascade()
        print("Face cascade classifier loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Initialize the webcam (default camera, index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please ensure a camera is connected.")
        return
    
    print("Webcam opened successfully.")
    print("Starting face detection with ellipse region filtering...\n")
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # Recognition variables
    match_found = False
    last_recognition_time = 0
    recognition_interval = 0.5  # Perform recognition every 0.5 seconds to reduce CPU load
    
    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Define ellipse parameters (narrower vertical ellipse at center)
            ellipse_center = (frame_width // 2, frame_height // 2)
            # Ellipse axes: width and height (as full lengths, not half)
            # Using 25% of frame width and 60% of frame height for a narrower vertical ellipse
            ellipse_axes = (int(frame_width * 0.25), int(frame_height * 0.60))
            
            # Convert original frame to grayscale for face detection
            # (Detect on original frame before masking for better accuracy)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect all faces in the grayscale frame
            all_faces = detect_faces(gray, face_cascade)
            
            # Filter faces to only include those inside the ellipse
            # Convert full axes lengths to half-lengths for the filter function
            ellipse_half_axes = (ellipse_axes[0] // 2, ellipse_axes[1] // 2)
            filtered_faces = filter_faces_in_ellipse(
                all_faces,
                ellipse_center,
                ellipse_half_axes
            )
            
            # Determine if any faces were detected
            has_face_detected = len(filtered_faces) > 0
            
            # Perform face recognition if enabled and face is detected
            match_found = False
            if recognition_enabled and has_face_detected:
                current_time = time.time()
                # Perform recognition at intervals to reduce CPU load
                if current_time - last_recognition_time >= recognition_interval:
                    if len(filtered_faces) > 0:
                        # Extract face region from original frame (before masking)
                        face_rect = filtered_faces[0]
                        face_image = extract_face_region(frame, face_rect)
                        
                        if face_image is not None:
                            # Extract embedding from detected face
                            detected_embedding = extract_face_embedding(face_image)
                            
                            if detected_embedding is not None:
                                # Compare with stored embeddings
                                match_found = find_best_match(
                                    detected_embedding,
                                    stored_embeddings,
                                    threshold=0.45  # Cosine similarity threshold (0.45 = 45% similarity)
                                )
                    
                    last_recognition_time = current_time
            
            # Create ellipse mask to black out everything outside
            ellipse_mask = create_ellipse_mask(frame.shape, ellipse_center, ellipse_axes)
            
            # Apply mask to black out everything outside the ellipse
            frame = apply_ellipse_mask(frame, ellipse_mask)
            
            # Draw the detection ellipse (border stays white)
            frame = draw_detection_ellipse(frame, ellipse_center, ellipse_axes, has_face_detected)
            
            # Display "Match found" message if recognition match is found
            if match_found:
                frame = display_match_message(frame, ellipse_center, ellipse_axes)
            
            # Calculate FPS
            fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - fps_start_time
            
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = current_time
            
            # Display FPS at top-left corner
            fps_text = f'FPS: {fps:.1f}'
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),  # Yellow color for good visibility
                2
            )
            
            # Display recognition status (optional, can be removed if not needed)
            if recognition_enabled:
                status_text = "Recognition: ON" if has_face_detected else "Recognition: Ready"
                cv2.putText(
                    frame,
                    status_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),  # Light gray
                    1
                )
            
            # Display the frame in a window
            cv2.imshow('Attendify - Face Detection', frame)
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting application...")
                break
    
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
    
    finally:
        # Clean up: release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Application closed.")


if __name__ == "__main__":
    main()
