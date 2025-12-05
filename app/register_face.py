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
from pathlib import Path

# Use DeepFace with ArcFace (InsightFace) for proper face embeddings
# This produces 512-dimensional discriminative embeddings
DEEPFACE_AVAILABLE = False
DeepFace = None

# Suppress TensorFlow AVX warnings (they're just warnings, not errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, Exception) as e:
    DEEPFACE_AVAILABLE = False
    DeepFace = None
    print(f"ERROR: DeepFace not available: {e}")
    print("\nTo install TensorFlow for Apple Silicon Mac:")
    print("  1. For Apple Silicon (M1/M2/M3): pip install tensorflow-macos tensorflow-metal")
    print("  2. Then install: pip install deepface")
    print("\nNote: tensorflow-metal enables GPU acceleration on Apple Silicon.")


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
        # DeepFace handles: face detection, alignment, preprocessing, and normalization
        embedding_result = DeepFace.represent(
            face_rgb,
            model_name='ArcFace',  # InsightFace/ArcFace - 512D embeddings
            enforce_detection=True,  # Require face detection for quality
            align=True,  # Face alignment for better accuracy
            normalization='base'  # Base normalization (DeepFace handles preprocessing)
        )
        
        if embedding_result and len(embedding_result) > 0:
            embedding = np.array(embedding_result[0]['embedding'], dtype=np.float32)
            
            # L2 normalize the embedding (critical for proper cosine similarity)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                print("Warning: Zero-norm embedding detected.")
                return None
            
            return embedding
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting embedding: {e}")
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


def display_instruction(frame, instruction_text, sub_text="", ellipse_center=None, ellipse_axes=None, alpha=1.0):
    """
    Display instruction text on the frame with high-quality rendering and smooth transitions.
    
    Args:
        frame: The video frame to draw on
        instruction_text: Main instruction string
        sub_text: Optional sub-instruction string
        ellipse_center: Tuple (cx, cy) - center of the ellipse (for positioning)
        ellipse_axes: Tuple (a, b) - full lengths of the ellipse axes (for positioning)
        alpha: Float (0.0-1.0) - opacity for fade animation
    
    Returns:
        numpy.ndarray: Frame with text drawn
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Use high-quality font with better rendering
    font = cv2.FONT_HERSHEY_DUPLEX
    main_font_scale = 1.6  # Increased for sharper text
    sub_font_scale = 1.2   # Increased for sharper text
    font_thickness = 4     # Increased thickness for better visibility
    line_type = cv2.LINE_AA  # Anti-aliased lines for smooth, professional text
    
    # Position text to the right of the ellipse, outside of it
    if ellipse_center and ellipse_axes:
        # Calculate right edge of ellipse
        ellipse_right_edge = ellipse_center[0] + ellipse_axes[0] // 2
        # Position text with padding from ellipse edge
        text_x = ellipse_right_edge + 40
    else:
        # Fallback: position at 70% of frame width
        text_x = int(frame_width * 0.70)
    
    # Center text vertically
    text_y = frame_height // 2 - 40
    
    # Get text size for background rectangle
    text_size = cv2.getTextSize(instruction_text, font, main_font_scale, font_thickness)[0]
    
    # Ensure text doesn't go off screen
    max_text_width = frame_width - text_x - 20
    if text_size[0] > max_text_width:
        # Scale down font if needed
        main_font_scale = (max_text_width / text_size[0]) * main_font_scale
        text_size = cv2.getTextSize(instruction_text, font, main_font_scale, font_thickness)[0]
    
    # Calculate background rectangle coordinates
    bg_x1 = text_x - 15
    bg_y1 = text_y - 45
    bg_x2 = text_x + text_size[0] + 15
    bg_y2 = text_y + 15
    
    # Draw background with efficient alpha blending (no frame.copy() needed)
    if alpha >= 0.99:
        # Fully opaque - direct rectangle (fastest)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    else:
        # Semi-transparent - use ROI for efficient blending (much faster than frame.copy())
        bg_alpha = int(alpha * 200)
        roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
        overlay_rect = np.zeros_like(roi)
        overlay_rect[:] = (0, 0, 0)
        cv2.addWeighted(roi, 1.0 - (alpha * 0.8), overlay_rect, alpha * 0.8, 0, roi)
    
    # Draw main instruction with high-quality anti-aliasing
    text_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
    cv2.putText(
        frame,
        instruction_text,
        (text_x, text_y),
        font,
        main_font_scale,
        text_color,
        font_thickness,
        line_type  # Anti-aliased for sharp, professional text
    )
    
    # Draw sub-instruction if provided
    if sub_text:
        sub_text_size = cv2.getTextSize(sub_text, font, sub_font_scale, font_thickness)[0]
        sub_text_x = text_x
        sub_text_y = text_y + 70  # Increased spacing
        
        # Ensure sub-text doesn't go off screen
        if sub_text_size[0] > max_text_width:
            sub_font_scale = (max_text_width / sub_text_size[0]) * sub_font_scale
            sub_text_size = cv2.getTextSize(sub_text, font, sub_font_scale, font_thickness)[0]
        
        # Calculate sub-text background coordinates
        sub_bg_x1 = sub_text_x - 15
        sub_bg_y1 = sub_text_y - 30
        sub_bg_x2 = sub_text_x + sub_text_size[0] + 15
        sub_bg_y2 = sub_text_y + 15
        
        # Draw sub-text background with efficient blending
        if alpha >= 0.99:
            cv2.rectangle(frame, (sub_bg_x1, sub_bg_y1), (sub_bg_x2, sub_bg_y2), (0, 0, 0), -1)
        else:
            # Use ROI for efficient blending
            sub_roi = frame[sub_bg_y1:sub_bg_y2, sub_bg_x1:sub_bg_x2]
            sub_overlay = np.zeros_like(sub_roi)
            sub_overlay[:] = (0, 0, 0)
            cv2.addWeighted(sub_roi, 1.0 - (alpha * 0.8), sub_overlay, alpha * 0.8, 0, sub_roi)
        
        # Draw sub-text with high-quality anti-aliasing
        sub_text_color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
        cv2.putText(
            frame,
            sub_text,
            (sub_text_x, sub_text_y),
            font,
            sub_font_scale,
            sub_text_color,
            font_thickness,
            line_type  # Anti-aliased for sharp, professional text
        )
    
    return frame


def save_embedding(embedding, filename=None, embeddings_dir="embeddings"):
    """
    Save face embedding to a file in the embeddings directory.
    
    Args:
        embedding: numpy.ndarray - the face embedding vector
        filename: Optional string - custom filename. If None, generates timestamp-based name
        embeddings_dir: String - directory to save embeddings (default: "embeddings")
    
    Returns:
        str: The full path to the saved embedding file
    """
    # Create embeddings directory if it doesn't exist
    embeddings_path = Path(embeddings_dir)
    embeddings_path.mkdir(exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"student_embedding_{timestamp}.npy"
    
    # Ensure filename has .npy extension
    if not filename.endswith('.npy'):
        filename = filename + '.npy'
    
    # Create full path
    full_path = embeddings_path / filename
    
    # Save as .npy file (NumPy binary format)
    np.save(full_path, embedding)
    
    print(f"Embedding saved to: {full_path}")
    
    return str(full_path)


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
    
    # Initialize the webcam with optimized settings for high FPS
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please ensure a camera is connected.")
        return
    
    # Optimize camera settings for high FPS
    # Set buffer size to 1 to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Try to set high FPS (camera may limit this)
    cap.set(cv2.CAP_PROP_FPS, 120)
    # Set frame width and height for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam opened successfully.")
    print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Check DeepFace availability
    if not DEEPFACE_AVAILABLE:
        print("ERROR: DeepFace not available!")
        print("\nTo install TensorFlow for Apple Silicon Mac:")
        print("  1. pip install tensorflow-macos tensorflow-metal")
        print("  2. pip install deepface")
        print("\nNote: tensorflow-metal enables GPU acceleration on Apple Silicon.")
        cap.release()
        return
    
    print("Using DeepFace with ArcFace (InsightFace) model for face embeddings.")
    print("Note: Model will be downloaded automatically on first use (~100MB).")
    print("This produces 512-dimensional discriminative embeddings.\n")
    
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
    # Reduced for higher FPS - with 60+ FPS, 5 frames is ~0.08 seconds which is sufficient
    required_stable_frames = 5  # Need 5 consecutive frames with face detected
    
    # Animation state for smooth text transitions
    previous_instruction = ""
    current_instruction = ""
    animation_state = "fade_in"  # "fade_in", "stable", "fade_out"
    animation_start_time = None
    fade_duration = 0.3  # Reduced to 0.3 seconds for snappier, smoother transitions
    current_alpha = 0.0
    last_animation_time = None  # Track last animation update for smooth interpolation
    
    # Performance optimization: cache ellipse mask and skip frames for detection
    cached_ellipse_mask = None
    cached_frame_shape = None
    frame_counter = 0
    detection_skip_frames = 2  # Process face detection every N frames (2 = every other frame)
    last_detected_faces = []
    last_has_face_detected = False
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break
            
            # Mirror the frame horizontally to match FaceTime behavior (mirror-like view)
            # This makes left/right movements feel natural, like looking in a mirror
            frame = cv2.flip(frame, 1)
            
            frame_height, frame_width = frame.shape[:2]
            
            # Define ellipse parameters (narrower vertical ellipse at center)
            ellipse_center = (frame_width // 2, frame_height // 2)
            ellipse_axes = (int(frame_width * 0.25), int(frame_height * 0.60))
            
            # Performance: Store original frame before processing (needed for face extraction)
            # Only store when we might need it (when face is detected or getting close)
            frame_counter += 1
            
            # Skip face detection on some frames for better performance
            # Process detection every N frames, but use last result for skipped frames
            should_detect = (frame_counter % detection_skip_frames == 0)
            
            if should_detect:
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
                
                last_detected_faces = filtered_faces
                last_has_face_detected = len(filtered_faces) > 0
            else:
                # Use cached detection results
                filtered_faces = last_detected_faces
                last_has_face_detected = len(filtered_faces) > 0
            
            has_face_detected = last_has_face_detected
            
            # Store original frame only when we have a face (for potential extraction)
            # This avoids copying every frame when no face is present
            if has_face_detected:
                original_frame = frame.copy()
            else:
                original_frame = None
            
            # Check for stable face detection
            if has_face_detected:
                stable_face_count += 1
                if stable_face_count >= required_stable_frames:
                    face_detected_stable = True
            else:
                stable_face_count = 0
                face_detected_stable = False
            
            # Performance: Cache ellipse mask (only recreate if frame size changes)
            if cached_ellipse_mask is None or cached_frame_shape != frame.shape:
                cached_ellipse_mask = create_ellipse_mask(frame.shape, ellipse_center, ellipse_axes)
                cached_frame_shape = frame.shape
            
            # Apply cached mask
            frame = apply_ellipse_mask(frame, cached_ellipse_mask)
            
            # Draw ellipse
            frame = draw_detection_ellipse(frame, ellipse_center, ellipse_axes, has_face_detected)
            
            # Registration flow
            if current_step < len(registration_steps):
                instruction = registration_steps[current_step]
                
                # Handle instruction change and animation
                if instruction != current_instruction:
                    # Instruction changed - start fade out then fade in
                    if current_instruction != "":
                        # Start fade out of previous instruction
                        animation_state = "fade_out"
                        animation_start_time = time.time()
                    else:
                        # First instruction - start fade in
                        animation_state = "fade_in"
                        animation_start_time = time.time()
                    
                    previous_instruction = current_instruction
                    current_instruction = instruction
                
                # Initialize step timer
                if step_start_time is None:
                    step_start_time = time.time()
                
                elapsed = time.time() - step_start_time
                
                # Update animation state and alpha with smooth interpolation
                current_time = time.time()
                if animation_start_time is not None:
                    animation_elapsed = current_time - animation_start_time
                    
                    if animation_state == "fade_out":
                        # Smooth fade out with easing (ease-out curve for natural feel)
                        progress = min(1.0, animation_elapsed / fade_duration)
                        # Ease-out cubic for smoother transition
                        eased_progress = 1.0 - (1.0 - progress) ** 3
                        current_alpha = max(0.0, 1.0 - eased_progress)
                        
                        if current_alpha <= 0.0:
                            # Fade out complete, start fade in immediately
                            animation_state = "fade_in"
                            animation_start_time = current_time
                            current_alpha = 0.0
                    elif animation_state == "fade_in":
                        # Smooth fade in with easing (ease-in curve for natural feel)
                        progress = min(1.0, animation_elapsed / fade_duration)
                        # Ease-in cubic for smoother transition
                        eased_progress = progress ** 3
                        current_alpha = min(1.0, eased_progress)
                        
                        if current_alpha >= 1.0:
                            # Fade in complete, stay stable
                            animation_state = "stable"
                            current_alpha = 1.0
                    elif animation_state == "stable":
                        # Keep alpha at 1.0
                        current_alpha = 1.0
                else:
                    # Initialize animation
                    animation_state = "fade_in"
                    animation_start_time = current_time
                    current_alpha = 0.0
                
                last_animation_time = current_time
                
                # Display instruction with animation
                if face_detected_stable:
                    sub_text = "Face detected. Hold still..."
                else:
                    sub_text = "Please position your face in the ellipse."
                
                # Show previous instruction during fade out, then new instruction during fade in/stable
                if animation_state == "fade_out" and previous_instruction:
                    frame = display_instruction(
                        frame, 
                        previous_instruction, 
                        sub_text, 
                        ellipse_center, 
                        ellipse_axes, 
                        current_alpha
                    )
                else:
                    frame = display_instruction(
                        frame, 
                        current_instruction, 
                        sub_text, 
                        ellipse_center, 
                        ellipse_axes, 
                        current_alpha
                    )
                
                # Capture frame if face is stable and enough time has passed
                if face_detected_stable and elapsed >= step_wait_time:
                    if len(filtered_faces) > 0 and original_frame is not None:
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
                    # Average all collected embeddings (they're already normalized)
                    final_embedding = np.mean(embeddings_collected, axis=0)
                    
                    # Re-normalize the averaged embedding (important!)
                    norm = np.linalg.norm(final_embedding)
                    if norm > 0:
                        final_embedding = final_embedding / norm
                    
                    # Save embedding
                    filename = save_embedding(final_embedding)
                    print(f"\nFace registration completed!")
                    print(f"Embedding saved to: {filename}")
                    print(f"Total samples collected: {len(embeddings_collected)}")
                    
                    # Display completion message
                    completion_text = "Face registration completed."
                    exit_text = "Press 'q' to exit."
                    frame = display_instruction(frame, completion_text, exit_text, ellipse_center, ellipse_axes, 1.0)
                    
                    # Wait for 'q' key
                    cv2.imshow('Attendify - Face Registration', frame)
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    break
                else:
                    # Should not happen, but handle edge case
                    frame = display_instruction(frame, "Error: No embeddings collected.", "Press 'q' to exit.", ellipse_center, ellipse_axes, 1.0)
            
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

