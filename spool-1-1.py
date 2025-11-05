from ultralytics import YOLO
import numpy as np

def validate_wrist_position(person_kpts, min_percent, max_percent, wrist_type="both", max_shoulder_percent=30):
    """
    Validate if wrists are within specified vertical percentage range and shoulder width
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
        validation_results["shoulders"]["message"] = "Missing reference keypoints"
        return validation_results
    
    shoulder_avg_y = np.mean(shoulder_y_values)
    hip_avg_y = np.mean(hip_y_values)
    reference_range = hip_avg_y - shoulder_avg_y
    
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
            wrist_y = left_wrist[1].cpu().numpy()
            
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["left"]["vertical_percent"] = vertical_percent
            
            vertical_valid = min_percent <= vertical_percent <= max_percent
            validation_results["left"]["valid"] = vertical_valid
            
            validation_results["left"]["message"] = "Valid" if vertical_valid else f"vertical position {vertical_percent:.1f}% outside range [{min_percent}, {max_percent}]"
    
    # Validate right wrist (keypoint 9) - REMOVED HORIZONTAL VALIDATION
    if wrist_type in ["right", "both"] and len(person_kpts) > 9:
        right_wrist = person_kpts[9]
        if right_wrist[0] > 0 and right_wrist[1] > 0:
            wrist_y = right_wrist[1].cpu().numpy()
            
            vertical_percent = ((wrist_y - shoulder_avg_y) / reference_range) * 100
            validation_results["right"]["vertical_percent"] = vertical_percent
            
            vertical_valid = min_percent <= vertical_percent <= max_percent
            validation_results["right"]["valid"] = vertical_valid
            
            validation_results["right"]["message"] = "Valid" if vertical_valid else f"vertical position {vertical_percent:.1f}% outside range [{min_percent}, {max_percent}]"
    
    return validation_results

def analyze_pose_results(results, min_percent=20, max_percent=40, wrist_type="both", max_shoulder_percent=30):
    """
    Analyze pose detection results and validate wrist positions
    """
    analysis_results = []
    
    for i, result in enumerate(results):
        xy = result.keypoints.xy
        xyn = result.keypoints.xyn
        kpts = result.keypoints.data
        
        print(f"\n{'='*50}")
        print(f"Results for detection {i}:")
        print(f"Validation Range: Vertical [{min_percent}%, {max_percent}%], Shoulder width [0%, {max_shoulder_percent}%]")
        print(f"{'='*50}")
        
        person_analyses = []
        
        for person_idx, person_kpts in enumerate(xy):
            print(f"\n--- Person {person_idx + 1} Analysis ---")
            
            # Validate wrist positions and shoulder width
            validation_results = validate_wrist_position(
                person_kpts, 
                min_percent, 
                max_percent,
                wrist_type=wrist_type,
                max_shoulder_percent=max_shoulder_percent
            )
            
            # Calculate reference levels for reporting
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
                shoulder_distance = right_shoulder_x - left_shoulder_x
                
                print(f"Reference Levels:")
                print(f"  0% level (shoulders avg): Y = {shoulder_avg_y:.1f} px")
                print(f"  100% level (hips avg): Y = {hip_avg_y:.1f} px")
                print(f"  Reference range: {reference_range:.1f} px")
                print(f"  Shoulder distance: {shoulder_distance:.1f} px")
            
            # Display validation results
            print("\nShoulder Width Validation:")
            shoulders_results = validation_results["shoulders"]
            shoulders_status = "PASS" if shoulders_results["valid"] else "FAIL"
            print(f"  Shoulders: {shoulders_status} - {shoulders_results['message']}")
            print(f"    Horizontal distance: {shoulders_results['horizontal_distance']:.1f}% of reference range")
            
            print("\nWrist Position Validation:")
            
            # LEFT wrist results
            left_results = validation_results["left"]
            left_status = "PASS" if left_results["valid"] else "FAIL"
            print(f"  LEFT Wrist: {left_status} - {left_results['message']}")
            print(f"    Vertical: {left_results['vertical_percent']:.1f}%")
            
            # RIGHT wrist results  
            right_results = validation_results["right"]
            right_status = "PASS" if right_results["valid"] else "FAIL"
            print(f"  RIGHT Wrist: {right_status} - {right_results['message']}")
            print(f"    Vertical: {right_results['vertical_percent']:.1f}%")
            
            # Calculate and display detailed keypoint information
            print(f"\nDetailed Keypoint Information:")
            for kpt_idx, kpt in enumerate(person_kpts):
                x, y = kpt.cpu().numpy()
                if x > 0 and y > 0:
                    print(f"  Keypoint {kpt_idx}: X={x:.1f}, Y={y:.1f}")
            
            person_analyses.append({
                "person_id": person_idx + 1,
                "validation_results": validation_results,
                "reference_levels": {
                    "shoulder_avg_y": shoulder_avg_y if 'shoulder_avg_y' in locals() else 0,
                    "hip_avg_y": hip_avg_y if 'hip_avg_y' in locals() else 0,
                    "shoulder_boundaries": [left_shoulder_x, right_shoulder_x] if 'left_shoulder_x' in locals() else [0, 0]
                }
            })
        
        analysis_results.append({
            "detection_id": i,
            "persons": person_analyses
        })
    
    return analysis_results

def main():
    """
    Main function to run pose detection and wrist validation
    """
    # Load model
    model = YOLO("yolo11n-pose.pt")
    
    # Predict with the model
    image_url = r"C:\Users\RYZEN\Pictures\vlcsnap-2025-11-04-13h48m22s704.jpg"
    #image_url = "eu_website_product_headlines_110x140_0224_cc_update.jpg"
    print(f"Analyzing image: {image_url}")
    results = model(image_url)
    
    # Define validation parameters
    MIN_VERTICAL_PERCENT = -20
    MAX_VERTICAL_PERCENT = 30
    WRIST_TYPE = "both"
    MAX_SHOULDER_PERCENT = 20  # Maximum shoulder width as percentage of reference range
    
    # Analyze results
    analysis_results = analyze_pose_results(
        results, 
        min_percent=MIN_VERTICAL_PERCENT,
        max_percent=MAX_VERTICAL_PERCENT,
        wrist_type=WRIST_TYPE,
        max_shoulder_percent=MAX_SHOULDER_PERCENT
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_persons = 0
    total_valid_wrists = 0
    total_wrists_checked = 0
    total_valid_shoulders = 0
    
    for detection in analysis_results:
        for person in detection["persons"]:
            total_persons += 1
            validation = person["validation_results"]
            
            # Check shoulder validation
            if validation["shoulders"]["valid"]:
                total_valid_shoulders += 1
            
            for wrist_type in ["left", "right"]:
                if wrist_type in validation:
                    total_wrists_checked += 1
                    if validation[wrist_type]["valid"]:
                        total_valid_wrists += 1
    
    print(f"Total persons detected: {total_persons}")
    print(f"Total wrists checked: {total_wrists_checked}")
    print(f"Total valid wrists: {total_valid_wrists}")
    print(f"Total valid shoulder widths: {total_valid_shoulders}")
    print(f"Validation criteria: Vertical [{MIN_VERTICAL_PERCENT}%, {MAX_VERTICAL_PERCENT}%]")
    print(f"Shoulder width criteria: [0%, {MAX_SHOULDER_PERCENT}%] of reference range")
    
    return analysis_results

if __name__ == "__main__":
    analysis_results = main()