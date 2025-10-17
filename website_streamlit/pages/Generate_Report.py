import streamlit as st

# Session States
if 'generating_report' not in st.session_state:
    st.session_state['generating_report'] = False

# Functions
def GenerateReport(imageUploaded):
    st.write(st.session_state.generating_report)
    if imageUploaded is not None:
        st.session_state.generating_report = True
        for image in imageUploaded:
            print("hello")

# Main
uploadedImages = st.file_uploader("Upload Roof Image", type=["jpg", "jpeg"])

st.button("Generate", disabled=st.session_state.generating_report, on_click=GenerateReport(uploadedImages))
