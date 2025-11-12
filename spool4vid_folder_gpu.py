#this code is combinnation of test3-1.py and spool-1-1.py
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from datetime import datetime
import shutil
import torch

def validate_wrist_position(person_kpts, min_percent, max_percent, wrist_type="both", max_shoulder_percent=30):
    """
    Validate if wrists are within specified vertical percentage range and shoulder width
    This function is copied from spool-1-1.py
    """
    validation_results = {
        "left": {"valid": False, "vertical_percent": 0, "message": ""},
        "right": {"valid": False, "vertical_percent": 0, "message": ""},
        "shoulders": {"valid": False, "horizontal_distance": 0, "message": ""}
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
        validation_results["shoulders"]["message"] = "Missing reference keypoints"
        return validation_results
    
    shoulder_avg_y = np.mean(shoulder_y_values)
    hip_avg_y = np.mean(hip_y_values)
    reference_range = hip_avg_y - shoulder_avg_y
    
    if reference_range <= 0:
        validation_results["left"]["message"] = "Invalid reference range"
        validation_results["right"]["message"] = "Invalid reference range"
        validation_results["shoulders"]["message"] = "Invalid reference range"
        return validation_results
    
    # Get shoulder x positions and calculate horizontal distance
    left_shoulder_x = min(shoulder_x_values)
    right_shoulder_x = max(shoulder_x_values)
    shoulder_horizontal_distance = right_shoulder_x - left_shoulder_x
    
    # Validate shoulder horizontal distance
    shoulder_percent = (shoulder_horizontal_distance / reference_range) * 100
    validation_results["shoulders"]["horizontal_distance"] = shoulder_percent
    validation_results["shoulders"]["valid"] = 0 <= shoulder_percent <= max_shoulder_percent
    
    if not validation_results["shoulders"]["valid"]:
        validation_results["shoulders"]["message"] = f"Shoulder distance {shoulder_percent:.1f}% outside range [0, {max_shoulder_percent}]"
    else:
        validation_results["shoulders"]["message"] = "Valid"
    
    # Validate left wrist (keypoint 10) - REMOVED HORIZONTAL VALIDATION
    if wrist_type in ["left", "both"] and len(person_kpts) > 10:
        left_wrist = person_kpts[10]
        if left_wrist[0] > 0 and left_wrist[1] > 0:
            wrist_y = left_wrist[1]
            
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["left"]["vertical_percent"] = vertical_percent
            
            vertical_valid = min_percent <= vertical_percent <= max_percent
            validation_results["left"]["valid"] = vertical_valid
            
            validation_results["left"]["message"] = "Valid" if vertical_valid else f"vertical position {vertical_percent:.1f}% outside range [{min_percent}, {max_percent}]"
    
    # Validate right wrist (keypoint 9) - REMOVED HORIZONTAL VALIDATION
    if wrist_type in ["right", "both"] and len(person_kpts) > 9:
        right_wrist = person_kpts[9]
        if right_wrist[0] > 0 and right_wrist[1] > 0:
            wrist_y = right_wrist[1]
            
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["right"]["vertical_percent"] = vertical_percent
            
            vertical_valid = min_percent <= vertical_percent <= max_percent
            validation_results["right"]["valid"] = vertical_valid
            
            validation_results["right"]["message"] = "Valid" if vertical_valid else f"vertical position {vertical_percent:.1f}% outside range [{min_percent}, {max_percent}]"
    
    return validation_results

def setup_video_source(source):
    """Setup video capture based on input source"""
    cap = cv2.VideoCapture(source)
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
    valid_color = (0, 255, 0)  # Green for valid
    invalid_color = (0, 0, 255)  # Red for invalid
    
    # Draw for each person detected
    for person_idx, person_keypoints in enumerate(keypoints):
        # Draw lines (skeleton)
        for start_idx, end_idx in skeleton:
            if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints) and
                not np.isnan(person_keypoints[start_idx][0]) and 
                not np.isnan(person_keypoints[end_idx][0])):
                
                start_point = tuple(map(int, person_keypoints[start_idx]))
                end_point = tuple(map(int, person_keypoints[end_idx]))
                cv2.line(image, start_point, end_point, line_color, 2)
        
        # Draw keypoints and validation status
        for i, (x, y) in enumerate(person_keypoints):
            if not np.isnan(x) and not np.isnan(y):
                point = (int(x), int(y))
                # Draw circle for keypoint
                cv2.circle(image, point, 4, point_color, -1)
        
        # Draw validation results if available
        if validation_results and person_idx < len(validation_results):
            person_results = validation_results[person_idx]
            
            # Draw shoulder validation
            shoulder_y = int(np.mean([person_keypoints[5][1], person_keypoints[6][1]])) if len(person_keypoints) > 6 else 30
            shoulder_status = "PASS" if person_results["shoulders"]["valid"] else "FAIL"
            shoulder_color = valid_color if person_results["shoulders"]["valid"] else invalid_color
            cv2.putText(image, f"Shoulders: {shoulder_status}", (10, 30 + person_idx * 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, shoulder_color, 2)
            cv2.putText(image, f"Dist: {person_results['shoulders']['horizontal_distance']:.1f}%", 
                       (10, 55 + person_idx * 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, shoulder_color, 1)
            
            # Draw wrist validation
            if "left" in person_results and person_results["left"]["vertical_percent"] != 0:
                left_status = "PASS" if person_results["left"]["valid"] else "FAIL"
                left_color = valid_color if person_results["left"]["valid"] else invalid_color
                cv2.putText(image, f"Left Wrist: {left_status}", (10, 80 + person_idx * 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
                cv2.putText(image, f"Vert: {person_results['left']['vertical_percent']:.1f}%", 
                           (10, 105 + person_idx * 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
            
            if "right" in person_results and person_results["right"]["vertical_percent"] != 0:
                right_status = "PASS" if person_results["right"]["valid"] else "FAIL"
                right_color = valid_color if person_results["right"]["valid"] else invalid_color
                cv2.putText(image, f"Right Wrist: {right_status}", (200, 80 + person_idx * 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
                cv2.putText(image, f"Vert: {person_results['right']['vertical_percent']:.1f}%", 
                           (200, 105 + person_idx * 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)

def save_spool_pose_clip(source, start_frame, end_frame, base_filename, save_folder="spool_pose_clips"):
    """Save a video clip when spool pose is detected"""
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Create a new video capture for the source
    clip_cap = cv2.VideoCapture(source)
    
    # Get video properties
    frame_width = int(clip_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(clip_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(clip_cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        fps = 30
    
    # Set up video writer for the clip
    clip_filename = f"{base_filename}.mp4"
    clip_filepath = os.path.join(save_folder, clip_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_writer = cv2.VideoWriter(clip_filepath, fourcc, fps, (frame_width, frame_height))
    
    # Set the starting frame position
    clip_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and write frames from start_frame to end_frame
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = clip_cap.read()
        if not ret:
            break
        clip_writer.write(frame)
    
    # Release resources
    clip_cap.release()
    clip_writer.release()
    
    #print(f"Spool pose clip saved: {clip_filepath} (frames {start_frame}-{end_frame})")

def save_spool_pose_frame(frame, frame_count, person_id, validation_results, keypoints, save_folder="spool_pose"):
    """Save frame with overlay result when spool pose is detected"""
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Create a copy of the frame to draw on
    frame_with_overlay = frame.copy()
    
    # Draw pose keypoints and validation results on the frame
    draw_pose_keypoints(frame_with_overlay, keypoints, [validation_results])
    
    # Add additional information text
    height, width = frame_with_overlay.shape[:2]
    cv2.putText(frame_with_overlay, f"Frame: {frame_count}", (width - 200, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_with_overlay, f"Person: {person_id}", (width - 200, height - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Generate filename with timestamp and frame info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spool_pose_{timestamp}_frame{frame_count}_person{person_id}.jpg"
    filepath = os.path.join(save_folder, filename)
    
    # Save the frame with overlay
    cv2.imwrite(filepath, frame_with_overlay)
    #print(f"Spool pose detected! Frame saved: {filepath}")
    
    # Return the base filename without extension for video clip creation
    base_filename = f"spool_pose_{timestamp}_frame{frame_count}_person{person_id}"
    return base_filename

def process_video(source, output_path, model, confidence_threshold=0.5, 
                  min_vertical_percent=-20, max_vertical_percent=30, 
                  wrist_type="both", max_shoulder_percent=20):
    """Process video file and save output with spool pose detection"""
    
    # Setup video capture
    cap = setup_video_source(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get total frames for reference
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    #out, frame_width, frame_height, fps = create_video_writer(cap, output_path)
    
    print(f"Processing video: {source}")
    print(f"Output: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"Spool pose validation parameters:")
    print(f"  Vertical range: [{min_vertical_percent}%, {max_vertical_percent}%]")
    print(f"  Wrist type: {wrist_type}")
    print(f"  Max shoulder percent: {max_shoulder_percent}%")
    print("Press 'q' to quit, 'p' to pause")
    
    frame_count = 0
    paused = False
    spool_pose_count = 0
    
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
            results = model(center_region, conf=confidence_threshold, verbose=False)
            
            # Store validation results for all persons in this frame
            frame_validation_results = []
            
            # Draw pose keypoints on the central region
            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy()  # x, y coordinates
                
                # Adjust keypoints coordinates to match the original frame
                adjusted_keypoints = []
                for person_kp in keypoints:
                    adjusted_person_kp = []
                    for x, y in person_kp:
                        if not np.isnan(x) and not np.isnan(y):
                            # Adjust x coordinate to account for the offset
                            adjusted_x = x + start_x
                            adjusted_person_kp.append([adjusted_x, y])
                        else:
                            adjusted_person_kp.append([x, y])
                    adjusted_keypoints.append(np.array(adjusted_person_kp))
                
                # Validate wrist positions for each person
                for person_idx, person_kpts in enumerate(adjusted_keypoints):
                    validation_results = validate_wrist_position(
                        person_kpts, 
                        min_vertical_percent, 
                        max_vertical_percent,
                        wrist_type=wrist_type,
                        max_shoulder_percent=max_shoulder_percent
                    )
                    
                    frame_validation_results.append(validation_results)
                    
                    # Check if this is a valid spool pose and save frame
                    if (validation_results["shoulders"]["valid"] and 
                        ((wrist_type == "both" and validation_results["left"]["valid"] and validation_results["right"]["valid"]) or
                         (wrist_type == "left" and validation_results["left"]["valid"]) or
                         (wrist_type == "right" and validation_results["right"]["valid"]))):
                        
                        base_filename = save_spool_pose_frame(frame, frame_count, person_idx + 1, validation_results, [person_kpts])
                        spool_pose_count += 1
                        
                        # Create video clip: rewind 300 frames and capture 600 frames total
                        clip_start_frame = max(0, frame_count - 300)
                        clip_end_frame = min(total_frames - 1, frame_count + 299)  # 600 frames total
                        
                        # Save the video clip
                        save_spool_pose_clip(source, clip_start_frame, clip_end_frame, base_filename)
                        
                        # Skip ahead to after the clip
                        if clip_end_frame < total_frames - 1:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_end_frame + 1)
                            frame_count = clip_end_frame
                
                # Draw adjusted keypoints and validation results on the original frame
                draw_pose_keypoints(frame, adjusted_keypoints, frame_validation_results)
            
            # Draw a rectangle to visualize the processing area
            cv2.rectangle(frame, (start_x, 0), (end_x, height), (0, 255, 255), 2)
            cv2.putText(frame, "Processing Area (Center)", (start_x + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display spool pose count
            cv2.putText(frame, f"Spool Poses Detected: {spool_pose_count}", (width - 300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Write frame to output video
            #out.write(frame)
            
            # Display frame with keypoints
            cv2.imshow('Pose Detection - Output (Center Area)', frame)
            
            frame_count += 1
            #print(f"Processed frame {frame_count}/{total_frames}, Spool poses: {spool_pose_count}", end='\r')
        
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
    #out.release()
    cv2.destroyAllWindows()
    #print(f"\nProcessing complete! Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total spool poses detected: {spool_pose_count}")

# Modified function to process all .mp4 files in a folder
def process_specific_sources():
    """Process all .mp4 files in a specific folder and move them to processed_files after completion"""
    
    # Define folders
    input_folder = r"C:\RecordDownload"  # Folder containing .mp4 files to process
    processed_folder = os.path.join(input_folder, "processed_files")  # Folder to move processed files
    
    # Create processed_files folder if it doesn't exist
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        print(f"Created processed_files folder: {processed_folder}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 0  # Use first GPU
    else:
        print("CUDA is not available. Using CPU.")
        device = 'cpu'
    
    # Load model and move to GPU if available
    model = YOLO("yolo11s-pose.pt")
    
    # Move model to GPU
    model.to(device)

    # Spool pose validation parameters
    MIN_VERTICAL_PERCENT = -20
    MAX_VERTICAL_PERCENT = 20
    WRIST_TYPE = "both"
    MAX_SHOULDER_PERCENT = 10
    
    # Get all .mp4 files in the input folder
    mp4_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print(f"No .mp4 files found in {input_folder}")
        return
    
    print(f"Found {len(mp4_files)} .mp4 files to process:")
    for file in mp4_files:
        print(f"  - {file}")
    
    # Process each .mp4 file
    for mp4_file in mp4_files:
        file_path = os.path.join(input_folder, mp4_file)
        
        # Move the file to processed_files folder first
        try:
            destination_path = os.path.join(processed_folder, mp4_file)
            shutil.move(file_path, destination_path)
            print(f"Moved {mp4_file} to {processed_folder}")
        except Exception as e:
            print(f"Error moving {mp4_file} to processed folder: {e}")
            continue  # Skip this file if move fails
        
        print(f"\n{'='*60}")
        print(f"Processing: {mp4_file}")
        print(f"{'='*60}")
        
        # Generate output filename
        output_filename = f"processed_{mp4_file}"
        output_path = os.path.join(input_folder, output_filename)
        
        # Process the video from the processed_files folder
        process_video(destination_path, output_path, model,
                     min_vertical_percent=MIN_VERTICAL_PERCENT,
                     max_vertical_percent=MAX_VERTICAL_PERCENT,
                     wrist_type=WRIST_TYPE,
                     max_shoulder_percent=MAX_SHOULDER_PERCENT)
    
    print(f"\n{'='*60}")
    print(f"All files processed successfully!")
    print(f"Processed files are in: {processed_folder}")
    print(f"{'='*60}")

import time
if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    print(f"Function started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Execute the function
    process_specific_sources()
    
    # Record finish time and calculate total time
    finish_time = time.time()
    total_time = finish_time - start_time
    
    print(f"Function finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time))}")
    print(f"Total execution time: {total_time:.2f} seconds")