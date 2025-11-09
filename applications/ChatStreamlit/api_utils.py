# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import streamlit as st
import requests

def _get_api_endpoint(name):
    return f"http://127.0.0.1:8000/{name}"


def get_api_response(question, session_id, model):

    headers = {
        "accept":"application/json",
        "Content-Type": "application/json"
    }

    data = {
        "question": question,
        "model": model,
    }

    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post(url=_get_api_endpoint("chat"), headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed. Error: {response.status_code} - {response.text}")
            return None
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def upload_document(file):
    print("Uploading file...")

    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(url=_get_api_endpoint("upload-doc"), files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


def list_documents():
    try:        
        response = requests.get(url=_get_api_endpoint("list-docs"))
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to load documents. Error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    

def delete_document(file_id: int):
    try:
        resp = requests.post(
            url=_get_api_endpoint("delete-doc"),
            json={"file_id": int(file_id)}
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Failed to delete document. Error: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None