#!/usr/bin/env python3
"""
Upload model to Hugging Face Hub
"""
import os
from huggingface_hub import HfApi, login

# Configuration
MODEL_REPO = "AryanJangde/legal-case-predictor-model"
MODEL_PATH = "content/ljp_legalbert_model"

def upload_model():
    print("🔐 Logging in to Hugging Face...")
    print("Enter your Hugging Face token (get it from https://huggingface.co/settings/tokens):")
    token = input("Token: ").strip()
    
    if not token:
        print("❌ Token is required!")
        return
    
    login(token=token)
    
    print(f"\n📤 Uploading model from {MODEL_PATH} to {MODEL_REPO}...")
    print("This may take several minutes...\n")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
        print(f"✅ Repository {MODEL_REPO} is ready")
    except Exception as e:
        print(f"⚠️  Note: {e}")
    
    # Upload files
    try:
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(f"\n✅ Model uploaded successfully!")
        print(f"📦 View at: https://huggingface.co/{MODEL_REPO}")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return
    
    print("\n🎉 Done! Update your API with:")
    print(f"   MODEL_DIR = '{MODEL_REPO}'")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model path not found: {MODEL_PATH}")
        exit(1)
    
    upload_model()

