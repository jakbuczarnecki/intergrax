# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import streamlit as st
from api_utils import upload_document, list_documents, delete_document


def display_sidebar():
    #Sidebar: Model selection component
    model_options = ["llama3.1:latest", "gpt-oss:20b"]
    st.sidebar.selectbox(label="Select model", options=model_options, key="model")

    #Sidebar: Upload document
    st.sidebar.header("Upload document")
    uploaded_file = st.sidebar.file_uploader(label="Choose a file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file) 
                if upload_response:
                    st.sidebar.success(f"File: {uploaded_file.name} uploaded successfully with ID {upload_response['file_id']}")           
                    st.session_state.documents = list_documents()

    #Sidebar: List documents
    st.sidebar.header("Uploaded documents")
    if st.sidebar.button("Refresh document list"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()
    

    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()


    documents = st.session_state.documents
    if documents:        
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']}, Uploaded: {doc['upload_timestamp']})")
        
        #Delete document option
        selected_file_id = st.sidebar.selectbox(
            label = "Select a document to delete", 
            options = [doc['id'] for doc in documents],
            format_func = lambda x: next(doc['filename'] for doc in documents if doc['id'] == x)
        )

        if st.sidebar.button("Delete selected document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Document with ID {selected_file_id} deleted successfully.")
                    st.session_state.documents = list_documents()
                else:
                    st.sidebar.error(f"Failed to delete document with ID {selected_file_id}.")
