import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd
import json
import time
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
PROJECT_ID = "your-project-id" # Update this
LOCATION = "us-central1"       # Keep this us-central1

# !!! FILL THESE IN AFTER TRAINING IS COMPLETE !!!
# Look for "Endpoint resource name" in the Console
MODEL_V1_ENDPOINT = "projects/..." 
MODEL_V2_ENDPOINT = "projects/..." 
# ---------------------

vertexai.init(project=PROJECT_ID, location=LOCATION)

def evaluate_version(endpoint_name, test_file_path, version_name):
    print(f"Evaluating {version_name}...")
    
    # Load the specific test set for this version
    with open(test_file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Connect to the tuned model
    model = GenerativeModel(endpoint_name)
    
    y_true = []
    y_pred = []
    
    print(f"Testing {len(data)} samples...")
    
    for i, entry in enumerate(data):
        # 1. Parse the input/output from JSONL
        # Structure: contents -> [user_part, model_part]
        user_input = entry['contents'][0]['parts'][0]['text']
        true_label = entry['contents'][1]['parts'][0]['text']
        
        try:
            # 2. Predict
            response = model.generate_content(user_input)
            pred_label = response.text.strip()
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            # Rate limiting safety
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")

    # 3. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"--- Accuracy for {version_name}: {acc:.4f} ---")
    return acc

if __name__ == "__main__":
    # Run comparison
    # We use the VALIDATION file for a quick check, or create a TEST file if you split it earlier
    acc_v1 = evaluate_version(MODEL_V1_ENDPOINT, "data/llm/val_v1.jsonl", "V1 (Raw)")
    acc_v2 = evaluate_version(MODEL_V2_ENDPOINT, "data/llm/val_v2.jsonl", "V2 (Descriptive)")

    print("\n==============================")
    print("FINAL RESULTS")
    print(f"V1 (Raw Data): {acc_v1}")
    print(f"V2 (Descriptive): {acc_v2}")
    print("==============================")
