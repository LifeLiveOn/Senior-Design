import streamlit as st
import tempfile
from cloud.upload_test import upload_to_bucket

BUCKET_NAME = "roofvision-images"

if 'generating_report' not in st.session_state:
    st.session_state.generating_report = False


def GenerateReport(imageUploaded):
    if not imageUploaded:
        st.warning("Please upload at least 1 image.")
        return

    st.session_state.generating_report = True

    urls = []

    for image in imageUploaded:
        # Create a temporary file so GCS upload function can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image.getbuffer())
            tmp_path = tmp.name

        # Upload to Google Cloud
        url = upload_to_bucket(BUCKET_NAME, tmp_path)

        if url:
            urls.append(url)
            st.success(f"Uploaded: {image.name}")
        else:
            st.error(f"Failed to upload {image.name}")

    st.write("Uploaded URLs:", urls)

    st.session_state.generating_report = False


st.title("Upload Roof Images to Cloud")

uploadedImages = st.file_uploader(
    "Upload Roof Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.button(
    "Generate",
    disabled=st.session_state.generating_report,
    on_click=GenerateReport,
    args=(uploadedImages,)
)
