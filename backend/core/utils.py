from google.cloud import storage
import uuid
import os


def upload_file_to_bucket(file_obj, bucket_name: str):
    """
    Upload a file object to a Google Cloud Storage bucket and return its public URL.
    Args:
        file_obj: The file object to upload.
        bucket_name: The name of the GCS bucket.
    Returns:
        The public URL of the uploaded file or None if upload failed.
    """
    if os.getenv("SKIP_CLOUD_UPLOAD") == "1":
        print("[INFO] SKIP_CLOUD_UPLOAD=1; skipping GCS upload for file object")
        return None

    try:
        client = storage.Client.from_service_account_json("key.json")
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print("GCS connection failed:", e)
        return None

    # generate a unique filename
    ext = os.path.splitext(file_obj.name)[1]
    unique_name = f"{uuid.uuid4()}{ext}"

    try:
        blob = bucket.blob(unique_name)

        # upload file content directly
        blob.upload_from_file(
            file_obj,
            content_type=file_obj.content_type,
            timeout=float(os.getenv("GCS_UPLOAD_TIMEOUT", "30")),
        )

        # PUBLIC URL - valid for buckets with Uniform Access
        public_url = f"https://storage.googleapis.com/{bucket_name}/{unique_name}"

        return public_url

    except Exception as e:
        print("Upload failed:", e)
        return None


def upload_local_file_to_bucket(local_path: str, bucket_name: str):
    """
    Upload a local file path to GCS and return the public URL.
    """
    from google.cloud import storage
    import uuid
    import os

    if os.getenv("SKIP_CLOUD_UPLOAD") == "1":
        print(
            "[INFO] SKIP_CLOUD_UPLOAD=1; skipping GCS upload for local file", local_path)
        return None

    try:
        client = storage.Client.from_service_account_json("key.json")
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print("GCS connection failed:", e)
        return None

    ext = os.path.splitext(local_path)[1]
    unique_name = f"{uuid.uuid4()}{ext}"

    try:
        blob = bucket.blob(unique_name)
        blob.upload_from_filename(
            local_path,
            timeout=float(os.getenv("GCS_UPLOAD_TIMEOUT", "30")),
        )
        public_url = f"https://storage.googleapis.com/{bucket_name}/{unique_name}"
        return public_url
    except Exception as e:
        print("Upload failed:", e)
        return None
