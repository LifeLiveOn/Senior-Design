#RoofVision Cloud Setup

This project connects our RoofVision app to **Google Cloud Storage** for uploading and storing roof images.

---

## How It Works
- Uses the `google-cloud-storage` Python library.  
- Uploads files to our GCP bucket: **`roofvision-images`**.  
- Each upload is authenticated using a private **`key.json`** service account file (not included for security).  
- Returns a cloud URL (`gs://...`) that the backend + ML model will use.

---

## How to Use
1. Add your own **`key.json`** file to this folder  
   _(do NOT commit it — it’s already ignored in `.gitignore`)_.

2. Install dependencies:
   ```bash
   pip install google-cloud-storage

3. Import the upload function in your backend:
**'from cloud.upload import upload_to_bucket'**

4. Call it with a file path:
**url = upload_to_bucket("roofvision-images", "/path/to/image.jpg")**

5. The function returns the cloud URL for the uploaded image

## FILES IN THIS FOLDER
upload.py - main upload function
README.md - this file
.gitignore - ignores key.json and venv/
