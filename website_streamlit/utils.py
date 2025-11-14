import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"


def refresh_access_token():
    refresh = st.session_state.get("refresh")
    if not refresh:
        return None

    res = requests.post(f"{API_BASE}/api/token/refresh/",
                        json={"refresh": refresh})
    if res.status_code == 200:
        new_access = res.json()["access"]
        st.session_state["access"] = new_access
        return new_access
    return None


def authorized_request(method, endpoint, **kwargs):
    access = st.session_state.get("access")
    headers = {"Authorization": f"Bearer {access}"}

    url = f"{API_BASE}{endpoint}"
    res = requests.request(method, url, headers=headers, **kwargs)

    if res.status_code == 401:
        new_access = refresh_access_token()
        if new_access:
            headers["Authorization"] = f"Bearer {new_access}"
            res = requests.request(method, url, headers=headers, **kwargs)

    return res
