import os
import time
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview.tuning import sft

## Load the environamental variables
load_dotenv()

# --- CONFIGURATION ---
LOCATION = os.getenv('LOCATION')
PROJECT_ID = os.getenv('PROJECT_ID')
BUCKET_NAME = os.getenv('BUCKET_NAME')

print(f'Project: {PROJECT_ID}')
print(f'Region: {LOCATION}')
print(f'Bucket: {BUCKET_NAME}')

# Initialize connection
aiplatform.init(project=PROJECT_ID, location=LOCATION)

def train_model(version):
    train_path = f'gs://{BUCKET_NAME}/week_10_data/train_{version}.jsonl'
    val_path = f'gs://{BUCKET_NAME}/week_10_data/val_{version}.jsonl'
    
    print(f'\n--- Submitting Training Job for {version} ---')
    
    try:
        # SFT using Gemini 1.5 Flash
        sft_tuning_job = sft.train(
            source_model = 'gemini-1.5-flash-001',
            train_dataset = train_path,
            validation_dataset = val_path,
            epochs = 4,
            tuned_model_display_name = f'iris-classifier-{version}',
        )
        print(f' Job submitted for {version}!')
        return sft_tuning_job
    
    except Exception as e:
        return f'Error submitting {version}: {e}'

print('Starting submission process...''')

## V1
job_v1 = train_model('v1')

time.sleep(60)

job_v2 = train_model('v2')

print('Jobs submitted! Visit the link below to track progress:')
print("https://console.cloud.google.com/vertex-ai/generative-ai/model-garden/tuning")
