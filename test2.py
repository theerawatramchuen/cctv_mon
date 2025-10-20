from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model(r"D:\Downloads\D24_20250528052000.mp4")  # predict on an image

# Access the results
for result in results:
    # Get the original image
    orig_img = result.orig_img
    
    # Create a copy to draw on
    output_img = orig_img.copy()
    
    # Get keypoints data
    keypoints = result.keypoints.xy.cpu().numpy()  # x, y coordinates
    
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
    
    # Draw skeleton connections
    for person_keypoints in keypoints:
        # Draw lines (skeleton)
        for start_idx, end_idx in skeleton:
            if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints) and
                not np.isnan(person_keypoints[start_idx][0]) and 
                not np.isnan(person_keypoints[end_idx][0])):
                
                start_point = tuple(map(int, person_keypoints[start_idx]))
                end_point = tuple(map(int, person_keypoints[end_idx]))
                cv2.line(output_img, start_point, end_point, line_color, 2)
        
        # Draw keypoints and labels
        for i, (x, y) in enumerate(person_keypoints):
            if not np.isnan(x) and not np.isnan(y):
                point = (int(x), int(y))
                # Draw circle for keypoint
                cv2.circle(output_img, point, 5, point_color, -1)
                # Add keypoint number/label
                #cv2.putText(output_img, str(i), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                # Add keypoint name
                #cv2.putText(output_img, keypoint_names[i], (point[0] + 10, point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    # Save the output image
    cv2.imwrite('pose_annotated_image.jpg', output_img)
    print("Annotated image saved as 'pose_annotated_image.jpg'")
    
    # Optional: Display the image (requires GUI)
    # cv2.imshow('Pose Detection', output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Print keypoints information
    print(f"Detected {len(keypoints)} person(s)")
    for i, person_keypoints in enumerate(keypoints):
        print(f"Person {i+1} keypoints:")
        for j, (x, y) in enumerate(person_keypoints):
            if not np.isnan(x) and not np.isnan(y):
                print(f"  {keypoint_names[j]}: ({x:.1f}, {y:.1f})")