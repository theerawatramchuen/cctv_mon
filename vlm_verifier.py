import requests
import json
import base64
from IPython.display import display, Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = 'http://10.151.28.9:11434/api/chat'
headers = {
    'Content-Type': 'application/json'
}

# Encode the image
image_path = r"zipping_pose\1761116844_valid_pose_frame_183339_persons_1.jpg"
encoded_image = encode_image(image_path)

# Display the image in the notebook
#display(Image(filename=image_path))

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

data = {
    "model": "gemma3:latest",#"llama3.2-vision:11b",#"gemma3:latest",#"minicpm-v:8b",
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

# Print the response content in a readable format
try:
    response_json = response.json()
    #print(json.dumps(response_json, indent=4))
except json.JSONDecodeError:
    print("Response is not valid JSON.")
    print(response.text)

import json
import re
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

# Usage
extracted_json = extract_json_content_robust(response_json)
print(extracted_json)


### model
""" 
Available models:
--------------------
• nomic-embed-text:latest
• scb10x/llama3.1-typhoon2-8b-instruct:latest
• gemma3:latest
• phi3:instruct
• llama3:latest
• dolphin-llama3:instruct
• minicpm-v:8b
• hf.co/nectec/Pathumma-llm-text-1.0.0:latest
• llama3.2-vision:11b
Updated 23-sep-25
""" 
###