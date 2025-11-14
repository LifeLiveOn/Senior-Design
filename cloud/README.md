# RoofVision Cloud Setup 

This project connects our RoofVision app to **Google Cloud Storage** for uploading and storing roof images.

## How It Works
- Uses the `google-cloud-storage` Python library.
- Uploads files to our GCP bucket: `roofvision-images`.
- Each upload is authenticated using a private `key.json` service account file (not shared).

## How to Use
1. Add your own **key.json** file (not included for security).
2. Install dependencies:
   ```bash
   pip install google-cloud-storage
