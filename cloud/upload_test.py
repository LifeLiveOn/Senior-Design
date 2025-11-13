from google.cloud import storage

def upload_to_bucket(bucket_name, file_path, destination_blob_name):
    client = storage.Client.from_service_account_json("key.json")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f" Uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    bucket_name = "roofvision-images"
    file_path = "test_image.jpg"
    destination_blob_name = "test_upload.jpg"
    upload_to_bucket(bucket_name, file_path, destination_blob_name)