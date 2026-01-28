from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import os

api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/vidrush"

print(f"Targeting Space: {repo_id}")

try:
    create_repo(repo_id, repo_type="space", space_sdk="gradio", private=False)
    print(f"Space created: {repo_id}")
except Exception as e:
    if "already exists" in str(e):
        print(f"Space already exists: {repo_id}")
    else:
        print(f"Error creating space: {e}")

# Files to upload
files_to_upload = ["app.py", "requirements.txt", "README.md"]

for file in files_to_upload:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="space"
        )

# Upload sample videos to make it usable out of the box
# I'll upload all .mp4 files in the root that aren't "vidrush_" outputs
videos = [f for f in os.listdir(".") if f.endswith(".mp4") and not f.startswith("vidrush_")]
# Limit to a few small ones if they are too heavy, but let's try to upload them
for v in videos:
    print(f"Uploading video: {v}...")
    try:
        upload_file(
            path_or_fileobj=v,
            path_in_repo=v,
            repo_id=repo_id,
            repo_type="space"
        )
    except Exception as e:
        print(f"Error uploading {v}: {e}")

print(f"Deployment complete! View your space at: https://huggingface.co/spaces/{repo_id}")
