
#"D:\Downloads\D24_20250528052000.mp4")  # predict on an image
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

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

def draw_pose_keypoints(image, keypoints):
    """Draw pose keypoints and skeleton on image"""
    # Define the skeleton connections for pose keypoints (COCO 17 keypoints format)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12),  # Torso connections
        (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
    ]
    
    # Define keypoint names for labeling (COCO format)
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
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

def process_video(source, output_path, model, confidence_threshold=0.5):
    """Process video or RTSP stream and save output"""
    
    # Setup video capture
    cap = setup_video_source(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Create video writer
    out, frame_width, frame_height, fps = create_video_writer(cap, output_path)
    
    print(f"Processing video: {source}")
    print(f"Output: {output_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")
    print("Press 'q' to quit, 'p' to pause")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            # Perform inference
            results = model(frame, conf=confidence_threshold)
            
            # Draw pose keypoints on frame
            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy()  # x, y coordinates
                draw_pose_keypoints(frame, keypoints)
            
            # Write frame to output video
            out.write(frame)
            
            # Display frame with keypoints
            cv2.imshow('Pose Detection - Output', frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}", end='\r')
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/Unpause
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f'frame_{frame_count}.jpg', frame)
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
    
    # Option 1: Process MP4 file
    # process_video(r"D:\Downloads\D24_20250528052000.mp4", "output_video.mp4", model)
    
    # Option 2: Process RTSP stream
    rtsp_url = "rtsp://admin:12345@10.153.61.163"
    process_video(rtsp_url, "rtsp_output.mp4", model)
    
    # Option 3: Process webcam (usually index 0)
    # process_video(0, "webcam_output.mp4", model)

if __name__ == "__main__":
    # Specific sources directly
    process_specific_sources()