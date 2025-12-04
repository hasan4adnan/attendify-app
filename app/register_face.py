"""
Attendify - Face Registration Module

This module performs face registration by capturing multiple face samples
from different angles and expressions, then generating a face embedding vector
that can be used for face recognition. The UI matches the detection module
with an ellipse-based detection zone.

Requirements:
    - Python 3.x
    - OpenCV (cv2) - install with: pip install opencv-python
    - DeepFace - install with: pip install deepface
    - NumPy - install with: pip install numpy

Usage:
    python register_face.py

Controls:
    - Press 'q' to quit the application
    - Follow on-screen instructions during registration

Note:
    On first run, DeepFace will automatically download the required model
    (InsightFace). This may take a few minutes.
    If you encounter TensorFlow/AVX errors, the program will use InsightFace
    which has better compatibility on Mac systems.
"""

import cv2
import time
import numpy as np
from datetime import datetime
import os

# Try face_recognition first (no TensorFlow dependency, better for Mac)
FACE_RECOGNITION_AVAILABLE = False
DEEPFACE_AVAILABLE = False
DeepFace = None

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    # Don't import DeepFace if face_recognition is available (avoids TensorFlow issues)
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    # Only try DeepFace if face_recognition is not available
    # Suppress TensorFlow AVX warnings before importing
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
    except (ImportError, Exception) as e:
        DEEPFACE_AVAILABLE = False
        DeepFace = None
        print(f"Warning: DeepFace not available: {e}")


def load_face_cascade():
    """
    Load the Haar Cascade classifier for face detection.
    
    Returns:
        cv2.CascadeClassifier: The face detection classifier
    """
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
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces


def is_point_inside_ellipse(point, center, axes):
    """
    Check if a point is inside an ellipse.
    
    Args:
        point: Tuple (x, y) - the point to check
        center: Tuple (cx, cy) - center of the ellipse
        axes: Tuple (a, b) - half-lengths of the ellipse axes
    
    Returns:
        bool: True if point is inside ellipse, False otherwise
    """
    x, y = point
    cx, cy = center
    a, b = axes
    
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
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
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
        0,
        0,
        360,
        255,
        -1
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
    mask_3channel = cv2.merge([mask, mask, mask])
    masked_frame = cv2.bitwise_and(frame, mask_3channel)
    return masked_frame


def draw_detection_ellipse(frame, center, axes, has_face_detected):
    """
    Draw the detection ellipse on the frame with color based on face detection.
    
    Args:
        frame: The video frame to draw on
        center: Tuple (cx, cy) - center of the ellipse
        axes: Tuple (a, b) - full lengths of the ellipse axes
        has_face_detected: Boolean - True if face is detected, False otherwise
    
    Returns:
        numpy.ndarray: Frame with ellipse drawn
    """
    if has_face_detected:
        ellipse_color = (0, 255, 0)  # Green (BGR)
    else:
        ellipse_color = (255, 255, 255)  # White (BGR)
    
    cv2.ellipse(
        frame,
        center,
        axes,
        0,
        0,
        360,
        ellipse_color,
        3  # Thick border
    )
    
    return frame


def extract_face_embedding(face_image):
    """
    Extract face embedding from a face image.
    Tries multiple methods for compatibility.
    
    Args:
        face_image: BGR image containing a face (numpy array)
    
    Returns:
        numpy.ndarray: Face embedding vector
    """
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Method 1: Try face_recognition library (uses dlib, better Mac compatibility)
    if FACE_RECOGNITION_AVAILABLE:
        try:
            # face_recognition expects RGB
            encodings = face_recognition.face_encodings(face_rgb)
            if encodings and len(encodings) > 0:
                return np.array(encodings[0])  # Returns 128-dimensional vector
        except Exception as e:
            print(f"Warning: face_recognition failed: {e}")
    
    # Method 2: Try DeepFace with ArcFace (only if face_recognition failed and DeepFace is available)
    if DEEPFACE_AVAILABLE and DeepFace is not None:
        try:
            embedding = DeepFace.represent(
                face_rgb,
                model_name='ArcFace',  # InsightFace/ArcFace - better Mac compatibility
                enforce_detection=False,
                align=True
            )
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
        except Exception as e1:
            # Fallback to VGG-Face if ArcFace fails
            try:
                embedding = DeepFace.represent(
                    face_rgb,
                    model_name='VGG-Face',
                    enforce_detection=False,
                    align=True
                )
                if embedding and len(embedding) > 0:
                    return np.array(embedding[0]['embedding'])
            except Exception as e2:
                print(f"Warning: DeepFace models failed: {e2}")
    
    # If all methods fail
    print("Error: Could not extract face embedding.")
    print("Please ensure either 'face_recognition' or 'deepface' is properly installed.")
    return None


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


def display_instruction(frame, instruction_text, sub_text=""):
    """
    Display instruction text on the frame.
    
    Args:
        frame: The video frame to draw on
        instruction_text: Main instruction string
        sub_text: Optional sub-instruction string
    
    Returns:
        numpy.ndarray: Frame with text drawn
    """
    frame_height = frame.shape[0]
    
    # Main instruction text (centered, larger)
    text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame_height // 2 - 30
    
    # Draw background rectangle for better readability
    cv2.rectangle(
        frame,
        (text_x - 10, text_y - 35),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0),
        -1
    )
    
    # Draw main instruction
    cv2.putText(
        frame,
        instruction_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),  # White text
        2
    )
    
    # Draw sub-instruction if provided
    if sub_text:
        sub_text_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        sub_text_x = (frame.shape[1] - sub_text_size[0]) // 2
        sub_text_y = text_y + 50
        
        cv2.rectangle(
            frame,
            (sub_text_x - 10, sub_text_y - 25),
            (sub_text_x + sub_text_size[0] + 10, sub_text_y + 10),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            sub_text,
            (sub_text_x, sub_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),  # Light gray text
            2
        )
    
    return frame


def save_embedding(embedding, filename=None):
    """
    Save face embedding to a file.
    
    Args:
        embedding: numpy.ndarray - the face embedding vector
        filename: Optional string - custom filename. If None, generates timestamp-based name
    
    Returns:
        str: The filename used to save the embedding
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"student_embedding_{timestamp}.npy"
    
    # Save as .npy file (NumPy binary format)
    np.save(filename, embedding)
    
    return filename


def main():
    """
    Main function to run the face registration application.
    """
    print("Initializing Attendify Face Registration Module...")
    print("Press 'q' to quit the application.\n")
    
    # Load the face cascade classifier
    try:
        face_cascade = load_face_cascade()
        print("Face cascade classifier loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please ensure a camera is connected.")
        return
    
    print("Webcam opened successfully.")
    
    # Check available embedding methods
    if FACE_RECOGNITION_AVAILABLE:
        print("Using face_recognition library (dlib-based, good Mac compatibility).")
    elif DEEPFACE_AVAILABLE:
        print("Using DeepFace library.")
        print("Note: Face embedding model will be downloaded automatically on first use.")
    else:
        print("ERROR: No face embedding library available!")
        print("Please install one of the following:")
        print("  pip install face-recognition  (recommended for Mac)")
        print("  pip install deepface")
        cap.release()
        return
    
    print("Starting face registration...")
    print("Please position your face inside the ellipse.\n")
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # Registration state
    registration_steps = [
        "Please look slightly up.",
        "Please look slightly down.",
        "Turn your head to the left.",
        "Turn your head to the right.",
        "Keep your face centered in the ellipse."
    ]
    
    current_step = 0
    embeddings_collected = []
    step_start_time = None
    step_wait_time = 2.0  # Wait 2 seconds before capturing each step
    face_detected_stable = False
    stable_face_count = 0
    required_stable_frames = 10  # Need 10 consecutive frames with face detected
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break
            
            frame_height, frame_width = frame.shape[:2]
            
            # Define ellipse parameters (narrower vertical ellipse at center)
            ellipse_center = (frame_width // 2, frame_height // 2)
            ellipse_axes = (int(frame_width * 0.25), int(frame_height * 0.60))
            
            # Keep original frame for face extraction (before masking)
            original_frame = frame.copy()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            all_faces = detect_faces(gray, face_cascade)
            
            # Filter faces inside ellipse
            ellipse_half_axes = (ellipse_axes[0] // 2, ellipse_axes[1] // 2)
            filtered_faces = filter_faces_in_ellipse(
                all_faces,
                ellipse_center,
                ellipse_half_axes
            )
            
            has_face_detected = len(filtered_faces) > 0
            
            # Check for stable face detection
            if has_face_detected:
                stable_face_count += 1
                if stable_face_count >= required_stable_frames:
                    face_detected_stable = True
            else:
                stable_face_count = 0
                face_detected_stable = False
            
            # Create and apply mask
            ellipse_mask = create_ellipse_mask(frame.shape, ellipse_center, ellipse_axes)
            frame = apply_ellipse_mask(frame, ellipse_mask)
            
            # Draw ellipse
            frame = draw_detection_ellipse(frame, ellipse_center, ellipse_axes, has_face_detected)
            
            # Registration flow
            if current_step < len(registration_steps):
                instruction = registration_steps[current_step]
                
                # Initialize step timer
                if step_start_time is None:
                    step_start_time = time.time()
                
                elapsed = time.time() - step_start_time
                
                # Display instruction
                if face_detected_stable:
                    sub_text = "Face detected. Hold still..."
                else:
                    sub_text = "Please position your face in the ellipse."
                
                frame = display_instruction(frame, instruction, sub_text)
                
                # Capture frame if face is stable and enough time has passed
                if face_detected_stable and elapsed >= step_wait_time:
                    if len(filtered_faces) > 0:
                        # Extract face region from original frame (before masking) for better quality
                        face_rect = filtered_faces[0]
                        face_image = extract_face_region(original_frame, face_rect)
                        
                        if face_image is not None:
                            # Extract embedding
                            embedding = extract_face_embedding(face_image)
                            
                            if embedding is not None:
                                embeddings_collected.append(embedding)
                                print(f"Step {current_step + 1}/{len(registration_steps)} completed.")
                                
                                # Move to next step
                                current_step += 1
                                step_start_time = None
                                stable_face_count = 0
                                face_detected_stable = False
                            else:
                                print(f"Warning: Failed to extract embedding for step {current_step + 1}.")
            else:
                # All steps completed - compute final embedding
                if len(embeddings_collected) > 0:
                    # Average all collected embeddings
                    final_embedding = np.mean(embeddings_collected, axis=0)
                    
                    # Save embedding
                    filename = save_embedding(final_embedding)
                    print(f"\nFace registration completed!")
                    print(f"Embedding saved to: {filename}")
                    print(f"Total samples collected: {len(embeddings_collected)}")
                    
                    # Display completion message
                    completion_text = "Face registration completed."
                    exit_text = "Press 'q' to exit."
                    frame = display_instruction(frame, completion_text, exit_text)
                    
                    # Wait for 'q' key
                    cv2.imshow('Attendify - Face Registration', frame)
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    break
                else:
                    # Should not happen, but handle edge case
                    frame = display_instruction(frame, "Error: No embeddings collected.", "Press 'q' to exit.")
            
            # Calculate and display FPS
            fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - fps_start_time
            
            if elapsed_time >= 1.0:
                fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = current_time
            
            fps_text = f'FPS: {fps:.1f}'
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),  # Yellow
                2
            )
            
            # Display progress
            if current_step < len(registration_steps):
                progress_text = f'Step {current_step + 1}/{len(registration_steps)}'
                cv2.putText(
                    frame,
                    progress_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # Display the frame
            cv2.imshow('Attendify - Face Registration', frame)
            
            # Check for 'q' key press to quit (only before completion)
            if current_step < len(registration_steps):
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nRegistration cancelled by user.")
                    break
    
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Application closed.")


if __name__ == "__main__":
    main()

