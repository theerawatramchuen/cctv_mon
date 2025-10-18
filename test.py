from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
#results = model("https://gaible.com/wp-content/uploads/2024/08/L-3111-2-scaled.jpg")  # predict on an image
#results = model("https://a.storyblok.com/f/233931/1299x1654/d3e5a79be4/eu_website_product_headlines_110x140_0224_cc_update.jpg")  # predict on an image
results = model("https://sc04.alicdn.com/kf/Hc5ed328a057e431eaf6eae60d8bc76b4x.jpeg")

# Process and display results
for i, result in enumerate(results):
    # Get keypoints data
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    
    print(f"Results for detection {i}:")
    print(f"Absolute coordinates (xy): {xy}")
    print(f"Normalized coordinates (xyn): {xyn}")
    print(f"Full keypoints data: {kpts}")
    
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
        
        # Calculate reference levels
        shoulder_ref_points = [5, 6]  # Left and right shoulders
        hip_ref_points = [11, 12]     # Left and right hips
        
        # Calculate average Y for shoulders (0% reference)
        shoulder_y_values = []
        for kpt_idx in shoulder_ref_points:
            if (len(person_kpts) > kpt_idx and 
                person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
                shoulder_y_values.append(person_kpts[kpt_idx][1].cpu().numpy())
        
        # Calculate average Y for hips (100% reference)
        hip_y_values = []
        for kpt_idx in hip_ref_points:
            if (len(person_kpts) > kpt_idx and 
                person_kpts[kpt_idx][0] > 0 and person_kpts[kpt_idx][1] > 0):
                hip_y_values.append(person_kpts[kpt_idx][1].cpu().numpy())
        
        if shoulder_y_values and hip_y_values:
            shoulder_avg_y = np.mean(shoulder_y_values)
            hip_avg_y = np.mean(hip_y_values)
            reference_range = hip_avg_y - shoulder_avg_y
            
            print(f"Reference Levels:")
            print(f"  0% level (shoulders avg): Y = {shoulder_avg_y:.1f} px")
            print(f"  100% level (hips avg): Y = {hip_avg_y:.1f} px")
            print(f"  Reference range: {reference_range:.1f} px")
            
            # Draw reference lines
            cv2.line(output_img, (0, int(shoulder_avg_y)), (img_width, int(shoulder_avg_y)), 
                    (0, 255, 255), 2)  # Yellow line for 0% level
            cv2.line(output_img, (0, int(hip_avg_y)), (img_width, int(hip_avg_y)), 
                    (255, 255, 0), 2)  # Cyan line for 100% level
            
            # Add reference level labels
            cv2.putText(output_img, "0% Level (Shoulders)", (10, int(shoulder_avg_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output_img, "100% Level (Hips)", (10, int(hip_avg_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            print("Cannot calculate reference levels - missing keypoints")
            shoulder_avg_y = 0
            hip_avg_y = 0
            reference_range = 1  # Avoid division by zero
        
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
                    # Normalize wrist position relative to shoulder-hip range
                    wrist_percentage = ((y1 - shoulder_avg_y) / reference_range) * 100
                    shoulder_percentage = ((y2 - shoulder_avg_y) / reference_range) * 100
                    
                    # Calculate relative percentage between wrist and shoulder
                    relative_percentage = wrist_percentage - shoulder_percentage
                else:
                    wrist_percentage = 0
                    shoulder_percentage = 0
                    relative_percentage = 0
                
                print(f"{description}:")
                print(f"  Keypoint {kpt1_idx} (wrist): {wrist_percentage:+.1f}% from shoulder level")
                print(f"  Keypoint {kpt2_idx} (shoulder): {shoulder_percentage:+.1f}% from shoulder level")
                print(f"  Wrist relative to shoulder: {relative_percentage:+.1f}%")
                print(f"  Absolute Y difference: {y1 - y2:+.1f} px")
                
                # Draw the connection line for these specific pairs
                point1 = tuple(person_kpts[kpt1_idx].cpu().numpy().astype(int))
                point2 = tuple(person_kpts[kpt2_idx].cpu().numpy().astype(int))
                cv2.line(output_img, point1, point2, (255, 0, 0), 3)  # Blue line for measured pairs
                
                # Add midpoint for percentage display
                mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
                percentage_text = f"{relative_percentage:+.1f}%"
                cv2.putText(output_img, percentage_text, 
                           (mid_point[0] + 10, mid_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
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
                elif kpt_idx in [9, 10]:  # Wrists (measured keypoints)
                    cv2.circle(output_img, point, 8, (255, 0, 0), -1)  # Blue for wrists
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
        "Vertical Analysis (0%=Shoulders, 100%=Hips):",
        "Yellow: Shoulders (Keypoints 5,6) - 0% reference",
        "Cyan: Hips (Keypoints 11,12) - 100% reference", 
        "Blue: Wrists (Keypoints 9,10) - measured points",
        "Percentage: Wrist position relative to shoulder level"
    ]
    
    for j, text in enumerate(legend_text):
        cv2.putText(output_img, text, (10, 30 + j * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(output_img, text, (10, 30 + j * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display the image
    cv2.imshow(f"Keypoints Detection with Vertical Analysis - Result {i}", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the output image
    output_filename = f"keypoints_vertical_analysis_{i}.jpg"
    cv2.imwrite(output_filename, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"Output image saved as: {output_filename}")