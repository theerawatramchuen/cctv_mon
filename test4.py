#this code is combinnation of test3-1.py and test-1-1.py
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

def validate_wrist_position(person_kpts, min_percent, max_percent, wrist_type="both"):
    """
    Validate if wrists are within specified vertical percentage range and horizontal position
    """
    validation_results = {
        "left": {"valid": False, "vertical_percent": 0, "horizontal_valid": False, "message": ""},
        "right": {"valid": False, "vertical_percent": 0, "horizontal_valid": False, "message": ""}
    }
    
    # Calculate reference levels
    shoulder_ref_points = [5, 6]
    hip_ref_points = [11, 12]
    
    # Calculate average Y for shoulders (0% reference)
    shoulder_y_values = []
    shoulder_x_values = []
    for kpt_idx in shoulder_ref_points:
        if (len(person_kpts) > kpt_idx and 
            person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
            shoulder_y_values.append(person_kpts[kpt_idx][1])
            shoulder_x_values.append(person_kpts[kpt_idx][0])
    
    # Calculate average Y for hips (100% reference)
    hip_y_values = []
    for kpt_idx in hip_ref_points:
        if (len(person_kpts) > kpt_idx and 
            person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
            hip_y_values.append(person_kpts[kpt_idx][1])
    
    if not shoulder_y_values or not hip_y_values:
        validation_results["left"]["message"] = "Missing reference keypoints"
        validation_results["right"]["message"] = "Missing reference keypoints"
        return validation_results
    
    shoulder_avg_y = np.mean(shoulder_y_values)
    hip_avg_y = np.mean(hip_y_values)
    reference_range = hip_avg_y - shoulder_avg_y
    
    if reference_range <= 0:
        validation_results["left"]["message"] = "Invalid reference range"
        validation_results["right"]["message"] = "Invalid reference range"
        return validation_results
    
    # Get shoulder x boundaries
    left_shoulder_x = min(shoulder_x_values)
    right_shoulder_x = max(shoulder_x_values)
    
    # Validate left wrist (keypoint 10)
    if wrist_type in ["left", "both"] and len(person_kpts) > 10:
        left_wrist = person_kpts[10]
        if left_wrist[0] > 0 and left_wrist[1] > 0:
            wrist_y = left_wrist[1]
            wrist_x = left_wrist[0]
            
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["left"]["vertical_percent"] = vertical_percent
            
            vertical_valid = min_percent <= vertical_percent <= max_percent
            horizontal_valid = left_shoulder_x <= wrist_x <= right_shoulder_x
            
            validation_results["left"]["valid"] = vertical_valid and horizontal_valid
            validation_results["left"]["horizontal_valid"] = horizontal_valid
            
            messages = []
            if not vertical_valid:
                messages.append(f"vertical position {vertical_percent:.1f}% outside range [{min_percent}, {max_percent}]")
            if not horizontal_valid:
                messages.append("horizontal position outside shoulder boundaries")
            
            validation_results["left"]["message"] = "Valid" if not messages else ", ".join(messages)
    
    # Validate right wrist (keypoint 9)
    if wrist_type in ["right", "both"] and len(person_kpts) > 9:
        right_wrist = person_kpts[9]
        if right_wrist[0] > 0 and right_wrist[1] > 0:
            wrist_y = right_wrist[1]
            wrist_x = right_wrist[0]
            
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["right"]["vertical_percent"] = vertical_percent
            
            vertical_valid = min_percent <= vertical_percent <= max_percent
            horizontal_valid = left_shoulder_x <= wrist_x <= right_shoulder_x
            
            validation_results["right"]["valid"] = vertical_valid and horizontal_valid
            validation_results["right"]["horizontal_valid"] = horizontal_valid
            
            messages = []
            if not vertical_valid:
                messages.append(f"vertical position {vertical_percent:.1f}% outside range [{min_percent}, {max_percent}]")
            if not horizontal_valid:
                messages.append("horizontal position outside shoulder boundaries")
            
            validation_results["right"]["message"] = "Valid" if not messages else ", ".join(messages)
    
    return validation_results

def draw_pose_keypoints(image, keypoints, validation_results=None):
    """Draw pose keypoints and skeleton on image with validation results"""
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
    valid_color = (0, 255, 0)  # Green for valid wrists
    invalid_color = (0, 0, 255)  # Red for invalid wrists
    
    # Draw for each person detected
    for person_idx, person_keypoints in enumerate(keypoints):
        # Draw lines (skeleton)
        for start_idx, end_idx in skeleton:
            if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints) and
                not np.isnan(person_keypoints[start_idx][0]) and 
                not np.isnan(person_keypoints[end_idx][0])):
                
                start_point = tuple(map(int, person_keypoints[start_idx]))
                end_point = tuple(map(int, person_keypoints[end_idx]))
                cv2.line(image, start_point, end_point, line_color, 1)
        
        # Draw keypoints and labels
        for i, (x, y) in enumerate(person_keypoints):
            if not np.isnan(x) and not np.isnan(y):
                point = (int(x), int(y))
                # Draw circle for keypoint
                cv2.circle(image, point, 1, point_color, -1)
        
        # Draw validation results if available
        if validation_results and person_idx < len(validation_results):
            person_validation = validation_results[person_idx]
            
            # Draw wrist validation status
            wrist_positions = {
                9: ("Right", person_validation["right"]),
                10: ("Left", person_validation["left"])
            }
            
            for kpt_idx, (side, result) in wrist_positions.items():
                if kpt_idx < len(person_keypoints):
                    x, y = person_keypoints[kpt_idx]
                    if not np.isnan(x) and not np.isnan(y):
                        color = valid_color if result["valid"] else invalid_color
                        status = "VALID" if result["valid"] else "INVALID"
                        
                        # Draw wrist circle with validation color
                        cv2.circle(image, (int(x), int(y)), 8, color, 1)
                        
                        # Add validation text
                        text = f"{side}: {status}"
                        text_x = int(x) + 15
                        text_y = int(y) - 10
                        #cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_valid_pose_frame_old(frame, frame_count, valid_persons_count):
    """Save frame with valid pose to zipping_pose folder"""
    # Create directory if it doesn't exist
    os.makedirs("zipping_pose", exist_ok=True)
    
    # Save frame
    filename = f"zipping_pose/valid_pose_frame_{frame_count}_persons_{valid_persons_count}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved valid pose frame: {filename}")
    return filename

import time

def save_valid_pose_frame(frame, frame_count, valid_persons_count):
    """Save frame with valid pose to zipping_pose folder"""
    # Create directory if it doesn't exist
    os.makedirs("zipping_pose", exist_ok=True)
    
    # Get current Unix timestamp
    timestamp = int(time.time())
    
    # Save frame with timestamp prefix
    filename = f"zipping_pose/{timestamp}_valid_pose_frame_{frame_count}_persons_{valid_persons_count}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved valid pose frame: {filename}")
    return filename

def process_video(source, output_path, model, confidence_threshold=0.5, 
                  min_vertical_percent=15, max_vertical_percent=40, wrist_type="both"):
    """Process video or RTSP stream and save output with wrist validation"""
    
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
    print(f"Wrist validation: Vertical [{min_vertical_percent}%, {max_vertical_percent}%], Type: {wrist_type}")
    print("Press 'q' to quit, 'p' to pause")
    
    frame_count = 0
    paused = False
    valid_pose_frames_saved = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            # Calculate the central 50% width area
            height, width = frame.shape[:2]
            start_x = int(width * 0.25)  # 25% from left
            end_x = int(width * 0.75)    # 75% from left (so width between is 50%)
            
            # Extract the central 50% region
            center_region = frame[:, start_x:end_x]
            
            # Perform inference only on the central region
            results = model(center_region, conf=confidence_threshold)
            
            # Initialize validation tracking
            all_validation_results = []
            valid_persons_in_frame = 0
            
            # Process results and validate wrists
            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy()  # x, y coordinates
                
                # Adjust keypoints coordinates to match the original frame
                adjusted_keypoints = []
                frame_validation_results = []
                
                for person_kp in keypoints:
                    adjusted_person_kp = []
                    person_validation = {"left": {"valid": False}, "right": {"valid": False}}
                    
                    for x, y in person_kp:
                        if not np.isnan(x) and not np.isnan(y):
                            # Adjust x coordinate to account for the offset
                            adjusted_x = x + start_x
                            adjusted_person_kp.append([adjusted_x, y])
                        else:
                            adjusted_person_kp.append([x, y])
                    
                    adjusted_keypoints.append(np.array(adjusted_person_kp))
                    
                    # Validate wrist positions for this person
                    if len(adjusted_person_kp) > 0:
                        validation = validate_wrist_position(
                            adjusted_person_kp, 
                            min_vertical_percent, 
                            max_vertical_percent,
                            wrist_type=wrist_type
                        )
                        frame_validation_results.append(validation)
                        
                        # Count valid persons
                        if wrist_type == "both":
                            if validation["left"]["valid"] and validation["right"]["valid"]:
                                valid_persons_in_frame += 1
                        elif wrist_type == "left" and validation["left"]["valid"]:
                            valid_persons_in_frame += 1
                        elif wrist_type == "right" and validation["right"]["valid"]:
                            valid_persons_in_frame += 1
                    
                    else:
                        frame_validation_results.append({"left": {"valid": False}, "right": {"valid": False}})
                
                all_validation_results.extend(frame_validation_results)
                
                # Draw adjusted keypoints on the original frame with validation results
                draw_pose_keypoints(frame, adjusted_keypoints, frame_validation_results)
            
            # Save frame if valid poses detected
            if valid_persons_in_frame > 0:
                saved_filename = save_valid_pose_frame(frame, frame_count, valid_persons_in_frame)
                valid_pose_frames_saved += 1
                
                # Add save confirmation text to frame
                cv2.putText(frame, f"SAVED: {saved_filename}", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw a rectangle to visualize the processing area
            cv2.rectangle(frame, (start_x, 0), (end_x, height), (0, 255, 255), 1)
            cv2.putText(frame, "Processing Area (Center)", (start_x + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add validation info to frame
            #cv2.putText(frame, f"Valid persons: {valid_persons_in_frame}", (10, 60), 
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv2.putText(frame, f"Validation: {min_vertical_percent}%-{max_vertical_percent}%", (10, 90), 
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame to output video
            #out.write(frame)
            
            # Display frame with keypoints
            cv2.imshow('Pose Detection - Output (Center Area)', frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}, Valid poses: {valid_persons_in_frame}, Saved: {valid_pose_frames_saved}", end='\r')
        
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
    print(f"Valid pose frames saved: {valid_pose_frames_saved} to 'zipping_pose' folder")

# Alternative: Direct usage without command line arguments
def process_specific_sources():
    """Process specific sources without command line arguments"""
    
    # Load model
    model = YOLO("yolo11s-pose.pt")
    
    # Wrist validation parameters
    MIN_VERTICAL_PERCENT = 15
    MAX_VERTICAL_PERCENT = 35
    WRIST_TYPE = "both"  # "both", "left", or "right"
    
    # Option 1: Process MP4 file
    # process_video(r"D:\Downloads\D24_20250528052000.mp4", "output_video.mp4", model,
    #               min_vertical_percent=MIN_VERTICAL_PERCENT,
    #               max_vertical_percent=MAX_VERTICAL_PERCENT,
    #               wrist_type=WRIST_TYPE)
    
    # Option 2: Process RTSP stream
    rtsp_url = "rtsp://admin:12345@10.153.62.88"
    process_video(rtsp_url, "rtsp_output.mp4", model,
                  min_vertical_percent=MIN_VERTICAL_PERCENT,
                  max_vertical_percent=MAX_VERTICAL_PERCENT,
                  wrist_type=WRIST_TYPE)
    
    # Option 3: Process webcam (usually index 0)
    # process_video(0, "webcam_output.mp4", model,
    #               min_vertical_percent=MIN_VERTICAL_PERCENT,
    #               max_vertical_percent=MAX_VERTICAL_PERCENT,
    #               wrist_type=WRIST_TYPE)

if __name__ == "__main__":
    # Specific sources directly
    process_specific_sources()