from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# Define the two ROIs (Region of Interest)
ROI_1 = (500, 100, 1400, 1080)  # Format: (x_start, y_start, x_end, y_end)
ROI_2 = (850, 50, 1020, 220)  # Format: (x_start, y_start, x_end, y_end)

def setup_video_source(source):
    """Setup video capture based on input source"""
    cap = cv2.VideoCapture(source)
    
    # Set buffer size to minimum for RTSP to reduce latency
    if source.startswith('rtsp://'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return cap

def create_video_writer(cap, output_path, fps=30):
    """Create video writer with same properties as input"""
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps if fps else int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:  # If FPS is 0, set a default
        fps = 30
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    return out, frame_width, frame_height, fps

def extract_roi(frame, roi_coords):
    """Extract a region of interest from a frame"""
    x_start, y_start, x_end, y_end = roi_coords
    return frame[y_start:y_end, x_start:x_end]

def apply_roi_to_frame(frame, processed_roi, roi_coords):
    """Apply processed ROI back to the original frame"""
    x_start, y_start, x_end, y_end = roi_coords
    frame[y_start:y_end, x_start:x_end] = processed_roi
    return frame

def draw_roi_boundaries(frame, roi_coords, roi_name, color):
    """Draw ROI boundaries and label with coordinates on the frame"""
    x_start, y_start, x_end, y_end = roi_coords
    
    # Draw rectangle around ROI
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
    
    # Draw filled background for text
    text = f"{roi_name}: ({x_start},{y_start})-({x_end},{y_end})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Position text above the ROI or below if ROI is at top
    if y_start - text_size[1] - 10 > 0:
        text_y = y_start - 10
    else:
        text_y = y_end + text_size[1] + 10
    
    # Draw background for text
    cv2.rectangle(frame, 
                 (x_start, text_y - text_size[1] - 5), 
                 (x_start + text_size[0] + 10, text_y + 5), 
                 color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x_start + 5, text_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_pose_keypoints(image, keypoints):
    """Draw pose keypoints and skeleton on image"""
    # Define the skeleton connections for pose keypoints (COCO 17 keypoints format)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12),  # Torso connections
        (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
    ]
    
    # Colors for visualization
    point_color = (0, 255, 0)  # Green for keypoints
    line_color = (255, 0, 0)   # Blue for skeleton lines
    text_color = (0, 0, 255)   # Red for text
    
    # Draw for each person detected
    for person_keypoints in keypoints:
        # Draw lines (skeleton)
        for start_idx, end_idx in skeleton:
            if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints) and
                not np.isnan(person_keypoints[start_idx][0]) and 
                not np.isnan(person_keypoints[end_idx][0])):
                
                start_point = tuple(map(int, person_keypoints[start_idx]))
                end_point = tuple(map(int, person_keypoints[end_idx]))
                cv2.line(image, start_point, end_point, line_color, 2)
        
        # Draw keypoints and labels
        for i, (x, y) in enumerate(person_keypoints):
            if not np.isnan(x) and not np.isnan(y):
                point = (int(x), int(y))
                # Draw circle for keypoint
                cv2.circle(image, point, 2, point_color, -1)
                # Add keypoint number
                #cv2.putText(image, str(i), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

def process_roi(roi_frame, model, confidence_threshold=0.5):
    """Process a single ROI and return the processed frame"""
    # Perform inference on the ROI
    results = model(roi_frame, conf=confidence_threshold)
    
    # Draw pose keypoints on the ROI
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # x, y coordinates
        draw_pose_keypoints(roi_frame, keypoints)
    
    return roi_frame

def process_video_with_rois(source, output_path, model, confidence_threshold=0.5):
    """Process video with two ROIs and combine results into single output"""
    
    # Setup video capture
    cap = setup_video_source(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Create video writer
    out, frame_width, frame_height, fps = create_video_writer(cap, output_path)
    
    print(f"Processing video with ROIs: {source}")
    print(f"ROI 1: {ROI_1}")
    print(f"ROI 2: {ROI_2}")
    print(f"Output: {output_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")
    print("Press 'q' to quit, 'p' to pause")
    
    frame_count = 0
    paused = False
    
    # Define colors for ROI boundaries
    roi1_color = (0, 255, 0)  # Green for ROI 1
    roi2_color = (0, 165, 255)  # Orange for ROI 2
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            # Create a copy of the original frame for processing
            processed_frame = frame.copy()
            
            # Extract ROIs from the frame
            roi1_frame = extract_roi(processed_frame, ROI_1)
            roi2_frame = extract_roi(processed_frame, ROI_2)
            
            # Process each ROI independently
            processed_roi1 = process_roi(roi1_frame, model, confidence_threshold)
            processed_roi2 = process_roi(roi2_frame, model, confidence_threshold)
            
            # Apply processed ROIs back to the processed frame
            # ROI_2 is applied last to take precedence in overlapping areas
            processed_frame = apply_roi_to_frame(processed_frame, processed_roi1, ROI_1)
            processed_frame = apply_roi_to_frame(processed_frame, processed_roi2, ROI_2)
            
            # Draw ROI boundaries and labels on the processed frame
            draw_roi_boundaries(processed_frame, ROI_1, "ROI 1", roi1_color)
            draw_roi_boundaries(processed_frame, ROI_2, "ROI 2", roi2_color)
            
            # Write processed frame to output video
            out.write(processed_frame)
            
            # Display frame with both ROIs processed and visible boundaries
            cv2.imshow('Pose Detection - Dual ROI Output', processed_frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}", end='\r')
        
        # Handle key presses
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/Unpause
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f'frame_{frame_count}.jpg', processed_frame)
            print(f"Frame saved as frame_{frame_count}.jpg")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete! Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")

# Alternative: Direct usage without command line arguments
def process_specific_sources():
    """Process specific sources without command line arguments"""
    
    # Load model
    model = YOLO("yolo11s-pose.pt")
    
    # Option 1: Process MP4 file with dual ROIs
    # process_video_with_rois(r"D:\Downloads\D24_20250528052000.mp4", "output_video.mp4", model)
    
    # Option 2: Process RTSP stream with dual ROIs
    # rtsp_url = "rtsp://username:password@ip:port/stream"
    rtsp_url = "rtsp://admin:12345@10.153.61.147"
    process_video_with_rois(rtsp_url, "rtsp_output.mp4", model)
    
    # Option 3: Process webcam (usually index 0) with dual ROIs
    # process_video_with_rois(0, "webcam_output.mp4", model)

if __name__ == "__main__":
    # Specific sources directly with dual ROI processing
    process_specific_sources()