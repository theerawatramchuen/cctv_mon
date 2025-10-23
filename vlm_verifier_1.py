import requests
import json
import base64
import os
import time
import shutil, re
from IPython.display import display, Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_content_robust(response_obj):
    content = response_obj["message"]["content"]
    
    # Use regex to extract JSON from markdown code blocks
    # This handles various markdown formats
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(1).strip()
    else:
        # If no code blocks found, use the entire content
        json_string = content.strip()
    
    # Parse the JSON
    extracted_data = json.loads(json_string)
    return extracted_data

url = 'http://10.151.28.9:11434/api/chat'
headers = {
    'Content-Type': 'application/json'
}

VLM_SYSTEM_PROMPT = (
    "You are a specialized CCTV security analyst focused on monitoring clean-room production environments. "
    "Your task is to detect and evaluate potential security breaches related to operator discipline. "
    
    "CRITICAL INSTRUCTION: IGNORE ALL VISUAL ANNOTATIONS\n"
    "- Disregard green pose estimation keypoints/dots\n"
    "- Ignore red and green circles around wrists\n"
    "- Treat annotations as non-existent for your analysis\n\n"

    "SPECIFIC FOCUS AREAS:\n"
    "1. UNZIPPING ACTIVITY: Detect if any operator is unzipping or partially opening their bunny suit\n"
    "2. CAMERA AWARENESS: Identify if any operator is deliberately looking at CCTV cameras\n"
    "3. HEADCOUNT: Count all personnel visible within yellow bounding boxes\n\n"
    
    "RESPONSE FORMAT:\n"
    "Respond ONLY with a JSON object containing these exact keys:\n"
    "- 'unzip_confidence' (1-5 scale)\n"
    "- 'looking_confidence' (1-5 scale) \n"
    "- 'headcount' (integer)\n\n"
    
    "SCORING CRITERIA:\n"
    "1: No suspicious activity detected\n"
    "2: Minimal/ambiguous activity\n" 
    "3: Moderate suspicion\n"
    "4: High confidence in violation\n"
    "5: Clear, deliberate security breach"
)

user_message = (
    "Analyze this clean-room CCTV footage and evaluate:\n"
    "1. Unzip Confidence: Likelihood of operator unzipping bunny suit (1-5)\n"
    "2. Looking Confidence: Likelihood of operator monitoring CCTV cameras (1-5)\n"
    "3. Headcount: Number of personnel in yellow bounding areas\n\n"
    
    "Base your assessment on:\n"
    "- Hand position and suit integrity\n"
    "- Gaze direction and camera awareness\n"
    "- Overall operator behavior patterns"
)

# Create confidence folders if they don't exist
for i in range(1, 6):
    folder_name = f"conf_{i}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def process_image(image_path):
    """Process a single image and return the unzip_confidence"""
    try:
        # Encode the image
        encoded_image = encode_image(image_path)
        
        # Display the image in the notebook (optional)
        display(Image(filename=image_path))

        data = {
            "model": "gemma3:latest",
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_message,
                    "images": [encoded_image]
                }
            ],
            "stream": False
        }

        response = requests.post(url, headers=headers, json=data)
        
        # Print the response status code
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                response_json = response.json()
                extracted_json = extract_json_content_robust(response_json)
                print(f"Analysis result: {extracted_json}")
                return extracted_json.get('unzip_confidence', 1)
            except json.JSONDecodeError:
                print("Response is not valid JSON.")
                print(response.text)
                return 1
            except Exception as e:
                print(f"Error processing response: {e}")
                return 1
        else:
            print(f"API request failed with status code: {response.status_code}")
            return 1
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return 1

# Main processing loop
while True:
    # Get list of images in zipping_pose folder
    image_folder = "zipping_pose"
    
    if not os.path.exists(image_folder):
        print(f"Folder '{image_folder}' not found. Creating it...")
        os.makedirs(image_folder)
        print("Waiting for images to process...")
        time.sleep(5)  # Wait 5 seconds before checking again
        continue
    
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print("No images found in zipping_pose folder. Waiting for new images...")
        time.sleep(5)  # Wait 5 seconds before checking again
        continue
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing: {image_file}")
        
        # Get confidence level from VLM
        confidence = process_image(image_path)
        
        # Ensure confidence is integer between 1-5
        confidence = max(1, min(5, int(confidence)))
        
        # Define target folder
        target_folder = f"conf_{confidence}"
        
        # Move image to confidence folder
        target_path = os.path.join(target_folder, image_file)
        shutil.move(image_path, target_path)
        print(f"Moved {image_file} to {target_folder}")
        
        # Small delay between processing images
        time.sleep(1)
    
    print("Batch processing completed. Waiting for new images...")