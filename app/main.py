"""
Attendify - Face Detection Module (Enhanced)

This module performs real-time face detection using OpenCV and Haar Cascade.
It opens the webcam, detects faces within a defined ellipse region, and provides
visual feedback through ellipse border color changes. Everything outside the 
ellipse is masked to black. Includes FPS display for performance monitoring.

Requirements:
    - Python 3.x
    - OpenCV (cv2) - install with: pip install opencv-python

Usage:
    python main.py

Controls:
    - Press 'q' to quit the application
"""

import cv2
import time
import numpy as np


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
    Draw the detection ellipse on the frame with color based on face detection.
    
    Args:
        frame: The video frame to draw on
        center: Tuple (cx, cy) - center of the ellipse
        axes: Tuple (a, b) - full lengths of the ellipse axes (for cv2.ellipse)
        has_face_detected: Boolean - True if face is detected, False otherwise
    
    Returns:
        numpy.ndarray: Frame with ellipse drawn
    """
    # Choose color based on face detection
    # Green when face detected, light/white when not
    if has_face_detected:
        ellipse_color = (0, 255, 0)  # Green (BGR)
    else:
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
        3      # Thicker border (increased from 2 to 3)
    )
    
    return frame


def main():
    """
    Main function to run the enhanced face detection application.
    """
    print("Initializing Attendify Face Detection Module (Enhanced)...")
    print("Press 'q' to quit the application.\n")
    
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
            
            # Create ellipse mask to black out everything outside
            ellipse_mask = create_ellipse_mask(frame.shape, ellipse_center, ellipse_axes)
            
            # Apply mask to black out everything outside the ellipse
            frame = apply_ellipse_mask(frame, ellipse_mask)
            
            # Draw the detection ellipse with color based on face detection
            frame = draw_detection_ellipse(frame, ellipse_center, ellipse_axes, has_face_detected)
            
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
            
            # Display the number of faces detected (inside ellipse)
            face_count = len(filtered_faces)
            if face_count > 0:
                cv2.putText(
                    frame,
                    f'Faces detected: {face_count}',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
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
