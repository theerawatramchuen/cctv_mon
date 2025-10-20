from ultralytics import YOLO
import cv2
import numpy as np

def validate_wrist_position(person_kpts, min_percent, max_percent, wrist_type="both"):
    """
    Validate if wrists are within specified vertical percentage range and horizontal position
    
    Args:
        person_kpts: Keypoints for a person
        min_percent: Minimum allowed vertical percentage
        max_percent: Maximum allowed vertical percentage  
        wrist_type: "left", "right", or "both"
    
    Returns:
        dict: Validation results for each wrist
    """
    validation_results = {
        "left": {"valid": False, "vertical_percent": 0, "horizontal_valid": False, "message": ""},
        "right": {"valid": False, "vertical_percent": 0, "horizontal_valid": False, "message": ""}
    }
    
    # Calculate reference levels
    shoulder_ref_points = [5, 6]  # Left and right shoulders
    hip_ref_points = [11, 12]     # Left and right hips
    
    # Calculate average Y for shoulders (0% reference)
    shoulder_y_values = []
    shoulder_x_values = []
    for kpt_idx in shoulder_ref_points:
        if (len(person_kpts) > kpt_idx and 
            person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
            shoulder_y_values.append(person_kpts[kpt_idx][1].cpu().numpy())
            shoulder_x_values.append(person_kpts[kpt_idx][0].cpu().numpy())
    
    # Calculate average Y for hips (100% reference)
    hip_y_values = []
    for kpt_idx in hip_ref_points:
        if (len(person_kpts) > kpt_idx and 
            person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
            hip_y_values.append(person_kpts[kpt_idx][1].cpu().numpy())
    
    if not shoulder_y_values or not hip_y_values:
        validation_results["left"]["message"] = "Missing reference keypoints"
        validation_results["right"]["message"] = "Missing reference keypoints"
        return validation_results
    
    shoulder_avg_y = np.mean(shoulder_y_values)
    hip_avg_y = np.mean(hip_y_values)
    reference_range = hip_avg_y - shoulder_avg_y
    
    # Get shoulder x boundaries
    left_shoulder_x = min(shoulder_x_values)  # Smaller x = left side
    right_shoulder_x = max(shoulder_x_values)  # Larger x = right side
    
    # Validate left wrist (keypoint 10)
    if wrist_type in ["left", "both"] and len(person_kpts) > 10:
        left_wrist = person_kpts[10]
        if left_wrist[0] > 0 and left_wrist[1] > 0:
            wrist_y = left_wrist[1].cpu().numpy()
            wrist_x = left_wrist[0].cpu().numpy()
            
            # Calculate vertical percentage
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["left"]["vertical_percent"] = vertical_percent
            
            # Check vertical position
            vertical_valid = min_percent <= vertical_percent <= max_percent
            
            # Check horizontal position (between shoulders)
            horizontal_valid = left_shoulder_x <= wrist_x <= right_shoulder_x
            
            validation_results["left"]["valid"] = vertical_valid and horizontal_valid
            validation_results["left"]["horizontal_valid"] = horizontal_valid
            
            # Generate message
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
            wrist_y = right_wrist[1].cpu().numpy()
            wrist_x = right_wrist[0].cpu().numpy()
            
            # Calculate vertical percentage
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["right"]["vertical_percent"] = vertical_percent
            
            # Check vertical position
            vertical_valid = min_percent <= vertical_percent <= max_percent
            
            # Check horizontal position (between shoulders)
            horizontal_valid = left_shoulder_x <= wrist_x <= right_shoulder_x
            
            validation_results["right"]["valid"] = vertical_valid and horizontal_valid
            validation_results["right"]["horizontal_valid"] = horizontal_valid
            
            # Generate message
            messages = []
            if not vertical_valid:
                messages.append(f"vertical position {vertical_percent:.1f}% outside range [{min_percent}, {max_percent}]")
            if not horizontal_valid:
                messages.append("horizontal position outside shoulder boundaries")
            
            validation_results["right"]["message"] = "Valid" if not messages else ", ".join(messages)
    
    return validation_results

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
#results = model("https://gaible.com/wp-content/uploads/2024/08/L-3111-2-scaled.jpg")  # predict on an image
results = model("https://a.storyblok.com/f/233931/1299x1654/d3e5a79be4/eu_website_product_headlines_110x140_0224_cc_update.jpg")  # predict on an image
#results = model("bunnysuit2.jpg")

# Define validation parameters - UPDATED VALUES
MIN_VERTICAL_PERCENT = 20  # Minimum allowed vertical percentage
MAX_VERTICAL_PERCENT = 40  # Maximum allowed vertical percentage
WRIST_TYPE = "both"        # Validate both wrists

# Process and display results
for i, result in enumerate(results):
    # Get keypoints data
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    
    print(f"Results for detection {i}:")
    print(f"Validation Range: Vertical [{MIN_VERTICAL_PERCENT}%, {MAX_VERTICAL_PERCENT}%], Horizontal [between shoulders], Wrist Type: {WRIST_TYPE}")
    
    # Get the original image
    orig_img = result.orig_img
    img_height, img_width = orig_img.shape[:2]
    
    # Draw keypoints and labels on the image
    output_img = orig_img.copy()
    
    # Define keypoint connections for skeleton
    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head and shoulders
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
        [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]  # Body and legs
    ]
    
    # Draw skeleton connections
    for person_idx, person_kpts in enumerate(xy):
        for connection in skeleton:
            start_idx, end_idx = connection
            if len(person_kpts) > max(start_idx, end_idx):
                start_point = tuple(person_kpts[start_idx].cpu().numpy().astype(int))
                end_point = tuple(person_kpts[end_idx].cpu().numpy().astype(int))
                
                # Only draw if both points are visible (confidence > 0)
                if (person_kpts[start_idx][0] > 0 and person_kpts[start_idx][1] > 0 and 
                    person_kpts[end_idx][0] > 0 and person_kpts[end_idx][1] > 0):
                    cv2.line(output_img, start_point, end_point, (0, 255, 0), 2)
    
    # Draw keypoints, labels, and calculate vertical percentages
    for person_idx, person_kpts in enumerate(xy):
        print(f"\n--- Person {person_idx + 1} Analysis ---")
        
        # Validate wrist positions with updated parameters
        validation_results = validate_wrist_position(
            person_kpts, 
            MIN_VERTICAL_PERCENT, 
            MAX_VERTICAL_PERCENT,
            wrist_type=WRIST_TYPE
        )
        
        # Calculate reference levels for visualization
        shoulder_ref_points = [5, 6]
        hip_ref_points = [11, 12]
        
        shoulder_y_values = []
        shoulder_x_values = []
        for kpt_idx in shoulder_ref_points:
            if (len(person_kpts) > kpt_idx and 
                person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
                shoulder_y_values.append(person_kpts[kpt_idx][1].cpu().numpy())
                shoulder_x_values.append(person_kpts[kpt_idx][0].cpu().numpy())
        
        hip_y_values = []
        for kpt_idx in hip_ref_points:
            if (len(person_kpts) > kpt_idx and 
                person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
                hip_y_values.append(person_kpts[kpt_idx][1].cpu().numpy())
        
        if shoulder_y_values and hip_y_values:
            shoulder_avg_y = np.mean(shoulder_y_values)
            hip_avg_y = np.mean(hip_y_values)
            reference_range = hip_avg_y - shoulder_avg_y
            
            left_shoulder_x = min(shoulder_x_values)
            right_shoulder_x = max(shoulder_x_values)
            
            print(f"Reference Levels:")
            print(f"  0% level (shoulders avg): Y = {shoulder_avg_y:.1f} px")
            print(f"  100% level (hips avg): Y = {hip_avg_y:.1f} px")
            print(f"  Shoulder boundaries: X = [{left_shoulder_x:.1f}, {right_shoulder_x:.1f}]")
            
            # Draw reference lines and boundaries
            cv2.line(output_img, (0, int(shoulder_avg_y)), (img_width, int(shoulder_avg_y)), 
                    (0, 255, 255), 2)  # Yellow line for 0% level
            cv2.line(output_img, (0, int(hip_avg_y)), (img_width, int(hip_avg_y)), 
                    (255, 255, 0), 2)  # Cyan line for 100% level
            
            # Draw validation range lines (20% and 40%)
            validation_min_y = shoulder_avg_y + (reference_range * MIN_VERTICAL_PERCENT / 100)
            validation_max_y = shoulder_avg_y + (reference_range * MAX_VERTICAL_PERCENT / 100)
            
            cv2.line(output_img, (0, int(validation_min_y)), (img_width, int(validation_min_y)), 
                    (0, 255, 0), 2)  # Green line for min validation level
            cv2.line(output_img, (0, int(validation_max_y)), (img_width, int(validation_max_y)), 
                    (0, 255, 0), 2)  # Green line for max validation level
            
            # Draw vertical boundaries (shoulder width)
            cv2.line(output_img, (int(left_shoulder_x), 0), (int(left_shoulder_x), img_height), 
                    (255, 0, 255), 1)  # Magenta line for left boundary
            cv2.line(output_img, (int(right_shoulder_x), 0), (int(right_shoulder_x), img_height), 
                    (255, 0, 255), 1)  # Magenta line for right boundary
            
            # Add reference level labels
            cv2.putText(output_img, "0% Level (Shoulders)", (10, int(shoulder_avg_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output_img, "100% Level (Hips)", (10, int(hip_avg_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(output_img, f"{MIN_VERTICAL_PERCENT}% Validation Min", (10, int(validation_min_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_img, f"{MAX_VERTICAL_PERCENT}% Validation Max", (10, int(validation_max_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_img, "Shoulder Boundaries", (int(left_shoulder_x) + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Display validation results
        print("\nWrist Position Validation:")
        for wrist, results in validation_results.items():
            status = "PASS" if results["valid"] else "FAIL"
            color = "GREEN" if results["valid"] else "RED"
            print(f"  {wrist.upper()} Wrist: {status} - {results['message']}")
            print(f"    Vertical: {results['vertical_percent']:.1f}%")
            print(f"    Horizontal: {'Within bounds' if results['horizontal_valid'] else 'Out of bounds'}")
        
        # Calculate vertical percentages for the specified keypoint pairs
        keypoint_pairs = [
            (9, 6, "Right Wrist to Right Shoulder"),  # Keypoint 9 to Keypoint 6
            (10, 5, "Left Wrist to Left Shoulder")    # Keypoint 10 to Keypoint 5
        ]
        
        for kpt1_idx, kpt2_idx, description in keypoint_pairs:
            if (len(person_kpts) > max(kpt1_idx, kpt2_idx) and 
                person_kpts[kpt1_idx][0] > 0 and person_kpts[kpt1_idx][1] > 0 and
                person_kpts[kpt2_idx][0] > 0 and person_kpts[kpt2_idx][1] > 0):
                
                y1 = person_kpts[kpt1_idx][1].cpu().numpy()  # Wrist keypoint
                y2 = person_kpts[kpt2_idx][1].cpu().numpy()  # Shoulder keypoint
                
                # Calculate vertical percentage based on predefined reference
                if reference_range > 0:
                    wrist_percentage = ((y1 - shoulder_avg_y) / reference_range) * 100
                    shoulder_percentage = ((y2 - shoulder_avg_y) / reference_range) * 100
                    relative_percentage = wrist_percentage - shoulder_percentage
                else:
                    wrist_percentage = 0
                    shoulder_percentage = 0
                    relative_percentage = 0
                
                # Draw the connection line for these specific pairs with color based on validation
                point1 = tuple(person_kpts[kpt1_idx].cpu().numpy().astype(int))
                point2 = tuple(person_kpts[kpt2_idx].cpu().numpy().astype(int))
                
                # Choose color based on validation result
                wrist_type = "right" if kpt1_idx == 9 else "left"
                is_valid = validation_results[wrist_type]["valid"]
                line_color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, Red if invalid
                
                cv2.line(output_img, point1, point2, line_color, 3)
                
                # Add midpoint for percentage display
                mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
                percentage_text = f"{relative_percentage:+.1f}%"
                cv2.putText(output_img, percentage_text, 
                           (mid_point[0] + 10, mid_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        
        # Draw all keypoints with numbers and coordinates
        for kpt_idx, kpt in enumerate(person_kpts):
            x, y = kpt.cpu().numpy()
            
            # Only draw if the keypoint is detected (confidence > 0)
            if x > 0 and y > 0:
                point = (int(x), int(y))
                
                # Draw keypoint circle with different colors
                if kpt_idx in [5, 6]:  # Shoulders
                    cv2.circle(output_img, point, 8, (0, 255, 255), -1)  # Yellow for shoulders
                elif kpt_idx in [11, 12]:  # Hips
                    cv2.circle(output_img, point, 8, (255, 255, 0), -1)  # Cyan for hips
                elif kpt_idx in [9, 10]:  # Wrists
                    # Color wrists based on validation
                    wrist_type = "right" if kpt_idx == 9 else "left"
                    is_valid = validation_results[wrist_type]["valid"]
                    circle_color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, Red if invalid
                    cv2.circle(output_img, point, 8, circle_color, -1)
                else:
                    cv2.circle(output_img, point, 5, (0, 0, 255), -1)  # Red for other keypoints
                
                # Add keypoint number and coordinates
                label = f"{kpt_idx}: ({int(x)},{int(y)})"
                cv2.putText(output_img, label, 
                           (point[0] + 10, point[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(output_img, label, 
                           (point[0] + 10, point[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add legend
    legend_text = [
        f"Wrist Validation (Vertical: [{MIN_VERTICAL_PERCENT}%, {MAX_VERTICAL_PERCENT}%])",
        "Yellow: Shoulders (Keypoints 5,6) - 0% reference",
        "Cyan: Hips (Keypoints 11,12) - 100% reference", 
        "Green Lines: Validation range boundaries",
        "Magenta: Shoulder horizontal boundaries",
        "Green: Valid wrist position",
        "Red: Invalid wrist position"
    ]
    
    for j, text in enumerate(legend_text):
        cv2.putText(output_img, text, (10, 30 + j * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(output_img, text, (10, 30 + j * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display the image
    cv2.imshow(f"Wrist Position Validation - Result {i}", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the output image
    output_filename = f"wrist_validation_{i}.jpg"
    cv2.imwrite(output_filename, output_img)# cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"Output image saved as: {output_filename}")