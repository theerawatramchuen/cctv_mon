from ultralytics import YOLO
import numpy as np

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
    left_shoulder_x = min(shoulder_x_values)
    right_shoulder_x = max(shoulder_x_values)
    
    # Validate left wrist (keypoint 10)
    if wrist_type in ["left", "both"] and len(person_kpts) > 10:
        left_wrist = person_kpts[10]
        if left_wrist[0] > 0 and left_wrist[1] > 0:
            wrist_y = left_wrist[1].cpu().numpy()
            wrist_x = left_wrist[0].cpu().numpy()
            
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
            wrist_y = right_wrist[1].cpu().numpy()
            wrist_x = right_wrist[0].cpu().numpy()
            
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

def analyze_pose_results(results, min_percent=20, max_percent=40, wrist_type="both"):
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
        print(f"Validation Range: Vertical [{min_percent}%, {max_percent}%], Horizontal [between shoulders]")
        print(f"{'='*50}")
        
        person_analyses = []
        
        for person_idx, person_kpts in enumerate(xy):
            print(f"\n--- Person {person_idx + 1} Analysis ---")
            
            # Validate wrist positions
            validation_results = validate_wrist_position(
                person_kpts, 
                min_percent, 
                max_percent,
                wrist_type=wrist_type
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
                
                print(f"Reference Levels:")
                print(f"  0% level (shoulders avg): Y = {shoulder_avg_y:.1f} px")
                print(f"  100% level (hips avg): Y = {hip_avg_y:.1f} px")
                print(f"  Shoulder boundaries: X = [{left_shoulder_x:.1f}, {right_shoulder_x:.1f}]")
            
            # Display validation results
            print("\nWrist Position Validation:")
            
            # LEFT wrist results
            left_results = validation_results["left"]
            left_status = "PASS" if left_results["valid"] else "FAIL"
            print(f"  LEFT Wrist: {left_status} - {left_results['message']}")
            print(f"    Vertical: {left_results['vertical_percent']:.1f}%")
            print(f"    Horizontal: {'Within bounds' if left_results['horizontal_valid'] else 'Out of bounds'}")
            
            # RIGHT wrist results  
            right_results = validation_results["right"]
            right_status = "PASS" if right_results["valid"] else "FAIL"
            print(f"  RIGHT Wrist: {right_status} - {right_results['message']}")
            print(f"    Vertical: {right_results['vertical_percent']:.1f}%")
            print(f"    Horizontal: {'Within bounds' if right_results['horizontal_valid'] else 'Out of bounds'}")
            
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
    image_url = "bunnysuit2.jpg"
    print(f"Analyzing image: {image_url}")
    results = model(image_url)
    
    # Define validation parameters
    MIN_VERTICAL_PERCENT = 15
    MAX_VERTICAL_PERCENT = 40
    WRIST_TYPE = "both"
    
    # Analyze results
    analysis_results = analyze_pose_results(
        results, 
        min_percent=MIN_VERTICAL_PERCENT,
        max_percent=MAX_VERTICAL_PERCENT,
        wrist_type=WRIST_TYPE
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_persons = 0
    total_valid_wrists = 0
    total_wrists_checked = 0
    
    for detection in analysis_results:
        for person in detection["persons"]:
            total_persons += 1
            validation = person["validation_results"]
            
            for wrist_type in ["left", "right"]:
                if wrist_type in validation:
                    total_wrists_checked += 1
                    if validation[wrist_type]["valid"]:
                        total_valid_wrists += 1
    
    print(f"Total persons detected: {total_persons}")
    print(f"Total wrists checked: {total_wrists_checked}")
    print(f"Total valid wrists: {total_valid_wrists}")
    print(f"Validation criteria: Vertical [{MIN_VERTICAL_PERCENT}%, {MAX_VERTICAL_PERCENT}%], Horizontal between shoulders")
    
    return analysis_results

if __name__ == "__main__":
    analysis_results = main()