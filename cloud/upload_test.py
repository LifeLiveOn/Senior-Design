import os
from google.cloud import storage
import uuid


def upload_to_bucket(bucket_name: str, file_path: str):
    #checks if file/path exists
    if not os.path.exists(file_path):
        print(f"couldn't find file: {file_path}")
        return None
    
    #check if file type is correct
    allowed = {".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(file_path)[1].lower()
    #notes on what this line is^:
        #extension = .....
        # splits the filename into two parts: [0] the name, [1] the extension
        #ex: "roof.jpg" -> ("roof", ".jpg")
        #[1] -> picks just the extension like .jpg
        #.lower() makes sure extension is lower case - JPG -> jpg
    
    if ext not in allowed:
        print(f"Warning '{ext}' is not a typical image extension. Uploading anyway...")

    # connect to GCS
    try:
        client = storage.Client.from_service_account_json("key.json")
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"failed to connect to Google Cloud: {e}")
        return None
    
    #generate safe + unique filename
    original_name = os.path.basename(file_path)
    unique_name = f"{uuid.uuid4()}_{original_name}"
    #notes for line ^
        # line creates a new filename that is guaranteed to be unique
        #uuid.uuid4() generates a random ID
        #made so no files overwrite each other when thousand of images uploaded

    #upload the file
    try:
        blob = bucket.blob(unique_name)
        blob.upload_from_filename(file_path)
        url = f"gs://{bucket_name}/{unique_name}"
        print(f"Uploaded: {original_name}")
        print(f"Cloud Location: {url}")
        return url
    
    except Exception as e:
        print(f"Upload failed for {file_path}: {e}")
        return None
    
#testing  
if __name__ == "__main__":
    bucket = "roofvision-images"
    test_file = "test.jpg"

    print("\nRunning local test..")
    upload_to_bucket(bucket,test_file)
    