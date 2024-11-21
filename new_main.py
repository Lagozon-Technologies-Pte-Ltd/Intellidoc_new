# ********************************************************************************************** #
# File name: newmain.py
# Created by: Satya
# Creation Date: 25-Jun-2024
# Application Name: INTELLIDOC_NEW.AI
#
# Change Details:
# Version No:     Date:        Changed by     Changes Done
# 01             25-Jun-2024   Satya          Initial Creation
# 01             04-Jul-2024   Satya          Added logic for HR and Legal  -Deployed over Azure
# 02             23-Jul-2024   Satya          Config file and path variables is set
# 03             01-Aug-2024   Satya          Package issue resolved -Deployed over Azure
# 04             17-Aug-2024   Satya          Hybrid Retriver
# 05             22-Aug-2024   Satya          Addition of more department and role with change in font
# 06             30-Aug-2024   Satya          Addition of unstructured.io and conversatinal chat with new UI
# 07             05-Sept-2024  Satya          Openai embedding and model with improved metadata extraction
# 08             09-sept-2024  Satya          multiple question implemented with fixed bugs
# 09             15-sept-2024  Satya          New prompts with gpt-4o-mini
# 10             20-sept-2024  Satya          Progress bar with numerical parameter in config file
# 11             26-sept-2024  Satya          Flag for demo
# ********************************************************************************************** #

# Added by Aruna for chromaDB SQLite version error
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Standard library imports
import os
import shutil
import time
import json
from datetime import datetime

# Third-party imports
import chromadb
import pandas as pd
import streamlit as st
from PIL import Image
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from streamlit_mic_recorder import speech_to_text
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig,
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import plotly.graph_objects as go

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from utils import *

# Local imports
from llama_parse import LlamaParse

load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TITLE = os.getenv("TITLE")
ROLE = os.getenv("ROLE").split(",")
PAGE = os.getenv("PAGE").split(",")
ASK_QUESTION = os.getenv("ASK_QUESTION")
ASK = os.getenv("ASK")
UPLOAD_DOC = os.getenv("UPLOAD_DOC")
E_QUESTION = os.getenv("E_QUESTION")
SECTION = os.getenv("SECTION").split(",")
DOCSTORE = os.getenv("DOCSTORE").split(",")
COLLECTION = os.getenv("COLLECTION").split(",")
DATABASE = os.getenv("DATABASE").split(",")
P_QUESTION = os.getenv("P_QUESTION")
INSERT_DOCUMENT = os.getenv("INSERT_DOCUMENT")
ADD_DOC = os.getenv("ADD_DOC")
DOC_ADDED = os.getenv("DOC_ADDED")
DELETE_DOC = os.getenv("DELETE_DOC")
C_DELETE = os.getenv("C_DELETE")
OUTPUT_FOLDER_FOR_UNSTRUCTURED = os.getenv("OUTPUT_FOLDER_FOR_UNSTRUCTURED")
INPUT_FILEPATH_FOR_UNSTRUCTURED = os.getenv("INPUT_FILEPATH_FOR_UNSTRUCTURED")
DOC_DELETED = os.getenv("DOC_DELETED")
N_DOC = os.getenv("N_DOC")

LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

BM25_TOP = os.getenv("BM25_TOP")
VEC_TOP = os.getenv("VEC_TOP")
TEMP_CHUNK_SIZE = os.getenv("TEMP_CHUNK_SIZE")
CHUNK_SIZE = os.getenv("CHUNK_SIZE")
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP")
BATCH_SIZE = os.getenv("BATCH_SIZE")
QA_PROMPT_STR = os.getenv("QA_PROMPT_STR")
LLM_INSTRUCTION = os.getenv("LLM_INSTRUCTION")
NO_METADATA=os.getenv("NO_METADATA")
METADATA_INSTRUCTION=os.getenv("METADATA_INSTRUCTION").split(",")

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_database = os.getenv("db_database")
db_port = os.getenv("db_port")

unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
unstructured_api_url = os.getenv("UNSTRUCTURED_API_URL")
Weight_of_importance_for_retriver = os.getenv("Weight_of_importance_for_retriver")



image = os.getenv("image")
ui_images = os.getenv("ui_images")
customer_self_demo_flag = os.getenv("customer_self_demo_flag")
show_chunk=os.getenv("show_chunk")
human_resources_documents_link = os.getenv("human_resources_documents_link").split(",")
legal_documents_link = os.getenv("legal_documents_link").split(",")
finance_documents_link = os.getenv("finance_documents_link").split(",")
operation_documents_link = os.getenv("operation_documents_link").split(",")
healthcare_documents_link = os.getenv("healthcare_documents_link").split(",")
insurance_documents_link = os.getenv("insurance_documents_link").split(",")
LD_documents_link = os.getenv("LD_documents_link").split(",")
others_documents_link = os.getenv("others_documents_link").split(",")
# AZURE_ACCOUNT_KEY=os.getenv("AZURE_ACCOUNT_KEY")
# AZURE_ACCOUNT_NAME=os.getenv("AZURE_ACCOUNT_NAME")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")


Settings.llm = OpenAI(model=LLM_MODEL, temperature=0)
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100,api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

openai_ef = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL
)

# CSV file to store query results
CSV_FILE_PATH = "record_results.csv"
gauge_config = {
    "Faithfulness": {"value": 95, "color": "green"},
    "Relevancy": {"value": 82, "color": "lightgreen"},
    "Precision": {"value": 80, "color": "yellow"},
    "Recall": {"value": 78, "color": "orange"},
    "Harmfulness": {"value": 15, "color": "green"}
}
def create_circular_gauge_chart(title, value, min_val, max_val, color, subtext):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20, 'color': 'black'}, },
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 1},
            'bgcolor': "white",
            'borderwidth': 0.7,
            'bordercolor': "black",

            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': subtext, 'font': {'size': 17, 'color': 'gray'}}
    ))
    fig.update_layout(width=100, height=120, margin=dict(t=50, b=20, l=20, r=20))
    return fig



# Function to append a row to the CSV file
def append_to_csv(database, question, context_str, response, metadata="No Metadata"):
    # Check if CSV exists and create a DataFrame
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
    else:
        df = pd.DataFrame(
            columns=["Database", "Question", "Context", "Response", "Metadata"]
        )

    # Create a new row
    new_row = pd.DataFrame(
        {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Database": [database],
            "Question": [question],
            "Context": [context_str],
            "Response": [response],
            "Metadata": [metadata],
        }
    )

    # Append the new row and save to CSV
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE_PATH, index=False)


def main():

    img = Image.open(image)
    st.set_page_config(
        page_title=TITLE, page_icon=img, layout="wide", initial_sidebar_state="expanded"
    )
    if customer_self_demo_flag:
        # Set up custom CSS for the SANDBOX INSTANCE label
        sandbox_css = """
            <style>
                .sandbox-instance {
                    background-color: #FFFFE0;  /* Lighter yellow background */
                    color: #00008B;  /* Dark blue text */

                    font-size: 16px;
                    padding: 8px 16px;
                    border-radius: 4px;
                    display: inline-block;
                    font-weight: bold;
                    position: absolute;
                    top: 53px;
                    right: 10px;
                    border: 1px solid black;  /* Thin black border */

                }
            </style>
        """

        # Use Streamlit's markdown to insert the custom CSS
        st.markdown(sandbox_css, unsafe_allow_html=True)

        # Displaying the label in the Streamlit app
        st.markdown(
            '<div class="sandbox-instance">SANDBOX INSTANCE</div>',
            unsafe_allow_html=True,
        )

    # Function to hide the sidebar with CSS
    def hide_sidebar():
        hide_style = """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """
        st.markdown(hide_style, unsafe_allow_html=True)

    # Function to style radio buttons horizontally
    def style_radio_buttons():
        style = """
        <style>
        .stRadio > div {
            display: flex;
            flex-direction: row;
            justify-content: left-align;
        }
        .stRadio > div > label {
        }
        </style>
        """
        st.markdown(style, unsafe_allow_html=True)

    # Apply the custom style for radio buttons
    style_radio_buttons()
    col1, col2 = st.columns([1, 9])

    with col1:
        st.image(ui_images, width=130)

    with col2:
        st.title(TITLE)
        # Use HTML to add spacing between the icon and title

    tabs = st.radio(
        "Choose your tab", ["**Admin**", "**User**"], label_visibility="hidden"
    )
    # Check which tab is selected
    if tabs == "**User**":

        # Show sidebar in Tab 1
        with st.sidebar:
            st.subheader("**Upload your temporary document**")
            # embed_model = OpenAIEmbedding()
            embed_model = Settings.embed_model

            # Define options without Markdown syntax
            parser_options = ["LlamaParse", "Unstructured.io"]

            # Create a selectbox for parser choice
            parser_temp_choice = st.selectbox(
                "Choose a parser:", parser_options, key="temp_select"
            )

            # Display the selected parser in bold
            st.markdown(f"**Selected Parser:** {parser_temp_choice}")
            uploaded_files = st.file_uploader(
                "**Choose your files**",
                type=["pdf", "csv", "xlsx", "docx", "pptx"],
                accept_multiple_files=True,
                disabled=bool(customer_self_demo_flag),
            )

            def initialize_index(uploaded_files, embed_model):
                temp_documents = []
                parsed_text = []

                # Parse files based on the chosen parser
                if parser_temp_choice == "LlamaParse":
                    for uploaded_file in uploaded_files:
                        file_content = uploaded_file.read()
                        temp_file_name = uploaded_file.name
                        result_text = use_llamaparse(file_content, temp_file_name)
                        parsed_text.append(result_text)
                else:
                    for uploaded_file in uploaded_files:
                        # Save each uploaded file temporarily
                        temp_file_path = INPUT_FILEPATH_FOR_UNSTRUCTURED
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Process the PDF and get output
                        result_text = use_unstructured(
                            temp_file_path, uploaded_file.name
                        )
                        parsed_text.append(result_text)

                # Split the parsed text into chunks of 1000 characters
                chunk_size = int(TEMP_CHUNK_SIZE)
                for text in parsed_text:
                    # Ensure the text is a string and split it into chunks
                    text_chunks = [
                        text[i : i + chunk_size]
                        for i in range(0, len(text), chunk_size)
                    ]
                    for chunk in text_chunks:
                        # Create document objects from each chunk
                        document = Document(text=chunk)
                        temp_documents.append(document)

                # Create the index from the chunked documents
                temp_index = VectorStoreIndex.from_documents(
                    temp_documents, embed_model=embed_model
                )
                return temp_index

            if uploaded_files:
                # Check if the index is already initialized
                if "temp_index" not in st.session_state:
                    if uploaded_files:
                        # Initialize the index and store it in session state
                        st.session_state.temp_index = initialize_index(
                            uploaded_files, embed_model
                        )
            else:
                # Clear the temp_index when documents are removed
                if "temp_index" in st.session_state:
                    del st.session_state["temp_index"]

        user_page()

    elif tabs == "**Admin**":
        # Hide sidebar in Tab 2
        # hide_sidebar()
        admin_page()


def admin_page():
    # Initialize session state for section if not already set
    if "section" not in st.session_state:
        st.session_state.section = SECTION[0]  # Default to the first section

    # Create a selectbox for section selection
    selected_section = st.selectbox(
        "**Select section**", SECTION, key="section_selectbox"
    )
    # Call admin_operations based on the selected section
    if selected_section in SECTION:
        index = SECTION.index(selected_section)
        admin_operations(COLLECTION[index], DATABASE[index])


def admin_operations(collection_name, db_path):

    # Initialize Streamlit app
    st.write(f"**Managing documents for collection: {collection_name}**")
    collection = init_chroma_collection(db_path, collection_name)
    if customer_self_demo_flag:
        warning_message = "1. Admin Persona to Insert/Delete the documents is disabled for “Sandbox-Try It Yourself” instance.\n\n2. Data Retrieval through Adhoc files upload under User Persona is disabled for “Sandbox-Try it Yourself” instance."

        placeholder = st.empty()  # Create a placeholder

        placeholder.warning(warning_message)

    # Initialize session state for document mapping and list
    if "doc_name_to_id" not in st.session_state:
        st.session_state.doc_name_to_id = {}
    if "doc_list" not in st.session_state:
        docs = collection.get()["metadatas"]
        ids = collection.get()["ids"]  # Get IDs separately to avoid multiple calls
        for doc_id, meta in zip(ids, docs):
            if "source" in meta:
                doc_name = meta["source"].split("\\")[-1]
                if doc_name not in st.session_state.doc_name_to_id:
                    st.session_state.doc_name_to_id[doc_name] = []
                st.session_state.doc_name_to_id[doc_name].append(
                    doc_id
                )  # Append all IDs
        st.session_state.doc_list = list(st.session_state.doc_name_to_id.keys())
    button_disabled = bool(customer_self_demo_flag)
    # print("value of flag",button_disabled)
    if st.button("**Show Document**", disabled=button_disabled):
        docs = collection.get()["metadatas"]
        ids = collection.get()["ids"]  # Get IDs separately
        st.session_state.doc_name_to_id = {}
        for doc_id, meta in zip(ids, docs):
            if "source" in meta:
                doc_name = meta["source"].split("\\")[-1]
                if doc_name not in st.session_state.doc_name_to_id:
                    st.session_state.doc_name_to_id[doc_name] = []
                st.session_state.doc_name_to_id[doc_name].append(
                    doc_id
                )  # Append all IDs
        st.session_state.doc_list = list(st.session_state.doc_name_to_id.keys())
        st.selectbox("**Documents**", st.session_state.doc_list, key="doc_select")

    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False

    if st.button("**Insert Document**", disabled=bool(customer_self_demo_flag)):
        st.session_state.show_uploader = True
    if st.session_state.show_uploader:

        # Define options without Markdown syntax
        parser_options = ["LlamaParse", "Unstructured.io"]

        # Create a selectbox for parser choice
        parser_choice = st.selectbox("**Choose a parser:**", parser_options)

        # Display the selected parser in bold
        st.markdown(f"**Selected Parser:** {parser_choice}")
        # File upload for adding documents

        files = st.file_uploader(
            "**Choose PDF files**", key="admin_upload", accept_multiple_files=True
        )

        if files and st.button("**Add Document**"):
            # Ensure that files is a list
            for file in files:
                file_content = file.read()
                file_name = file.name
                print("the name of file.....", file_name)
                # upload_to_blob_storage(
                #     AZURE_STORAGE_CONNECTION_STRING,
                #     AZURE_CONTAINER_NAME,
                #     collection_name,
                #     file_content,
                #     file_name,
                # )

                if parser_choice == "LlamaParse":
                    parsed_text = use_llamaparse(file_content, file_name)

                else:
                    file_path = INPUT_FILEPATH_FOR_UNSTRUCTURED
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    parsed_text = use_unstructured(file_path, file_name)
                    # print(parsed_text)

                base_splitter = SentenceSplitter(
                    chunk_size=int(CHUNK_SIZE), chunk_overlap=int(CHUNK_OVERLAP)
                )
                nodes = base_splitter.get_nodes_from_documents(
                    [Document(text=parsed_text)]
                )


                # Initialize storage context (by default it's in-memory)
                storage_context = StorageContext.from_defaults()

                # Use the file name as the base ID
                base_file_name = os.path.basename(file.name)
                chunk_ids = []
                metadatas = []

                # Prepare data for storage and collection
                for i, node in enumerate(nodes):
                    chunk_id = f"{base_file_name}_{i + 1}"  # Create chunk IDs like "file.pdf_1", "file.pdf_2", etc.
                    chunk_ids.append(chunk_id)

                    metadata = {"type": base_file_name, "source": file.name}
                    metadatas.append(metadata)

                    # Create a new Document or Node object with text and metadata
                    document = Document(text=node.text, metadata=metadata, id_=chunk_id)

                    # Store the document in the storage context
                    storage_context.docstore.add_documents([document])
                # Load existing documents from the .json file if it exists
                for i in range(len(DOCSTORE)):
                    if collection_name in DOCSTORE[i]:
                        coll = DOCSTORE[i]
                        break
                existing_documents = {}
                if os.path.exists(coll):
                    with open(coll, "r") as f:
                        existing_documents = json.load(f)

                    # Persist the storage context (if necessary)
                    storage_context.docstore.persist(coll)

                    # Load new data from the same file (or another source)
                    with open(coll, "r") as f:
                        st_data = json.load(f)

                    # Update existing_documents with st_data
                    for key, value in st_data.items():
                        if key in existing_documents:
                            # Ensure the existing value is a list before extending
                            if isinstance(existing_documents[key], list):
                                existing_documents[key].extend(
                                    value
                                )  # Merge lists if key exists
                            else:
                                # If it's not a list, you can choose to replace it or handle it differently
                                existing_documents[key] = (
                                    [existing_documents[key]] + value
                                    if isinstance(value, list)
                                    else [existing_documents[key], value]
                                )
                        else:
                            existing_documents[key] = value  # Add new key-value pair

                    merged_dict = {}
                    for d in existing_documents["docstore/data"]:
                        merged_dict.update(d)
                    final_dict = {}
                    final_dict["docstore/data"] = merged_dict

                    # Write the updated documents back to the JSON file
                    with open(coll, "w") as f:
                        json.dump(final_dict, f, indent=4)

                else:
                    # Persist the storage context if the file does not exist
                    storage_context.docstore.persist(coll)

                # embed_model =OpenAIEmbedding()
                embed_model = Settings.embed_model

                # Create the vector store index
                VectorStoreIndex(
                    nodes, storage_context=storage_context, embed_model=embed_model
                )

                # # Add the chunks to the collection
                # collection.add(
                #     documents=[node.text for node in nodes],
                #     metadatas=metadatas,
                #     ids=chunk_ids
                # )

                batch_size = int(BATCH_SIZE)
                for i in range(0, len(nodes), batch_size):
                    batch_nodes = nodes[i : i + batch_size]
                    try:
                        collection.add(
                            documents=[node.text for node in batch_nodes],
                            metadatas=metadatas[i : i + batch_size],
                            ids=chunk_ids[i : i + batch_size],
                        )
                        time.sleep(5)  # Add a retry with a delay

                    except:
                        # Handle rate limit by adding a delay or retry mechanism
                        print("Rate limit error has occurred at this moment")

                # Update session state
                if base_file_name not in st.session_state.doc_name_to_id:
                    st.session_state.doc_name_to_id[base_file_name] = (
                        []
                    )  # Initialize if not present
                st.session_state.doc_name_to_id[base_file_name].extend(
                    chunk_ids
                )  # Store all chunk IDs under the base file name
                st.session_state.doc_list.append(
                    base_file_name
                )  # Store only the base file name

            st.success("**Documents added successfully!**")
            st.session_state.show_uploader = False

    # Deletion of documents
    if "show_delete" not in st.session_state:
        st.session_state.show_delete = False

    if st.button("**Delete Document**", disabled=bool(customer_self_demo_flag)):
        st.session_state.show_delete = True

    if st.session_state.show_delete:
        # Ensure the document list is populated

        if not st.session_state.doc_list:
            st.warning("**No documents available to delete.**")
        else:
            selected_doc_to_delete = st.selectbox(
                "**Select Document to Delete**",
                st.session_state.doc_list,
                key="doc_delete_select",
            )
            # print("Selected document for deletion:", selected_doc_to_delete)

            if st.button("**Confirm Delete**"):
                if selected_doc_to_delete:
                    # delete_from_blob_storage(
                    #     AZURE_STORAGE_CONNECTION_STRING,
                    #     AZURE_CONTAINER_NAME,
                    #     selected_doc_to_delete,
                    #     collection_name
                    # )
                    # Retrieve all chunk IDs associated with the selected document name
                    ids_to_delete = st.session_state.doc_name_to_id.get(
                        selected_doc_to_delete, []
                    )
                    # print("IDs to delete:", ids_to_delete)

                    if ids_to_delete:
                        try:
                            current_ids = collection.get()["ids"]
                            # print("Current IDs in collection:", current_ids)

                            # Check for existing IDs to delete
                            ids_to_delete_existing = [
                                id for id in ids_to_delete if id in current_ids
                            ]
                            # print("Valid IDs to delete:", ids_to_delete_existing)


                            # Step 1: Read the JSON file
                            for i in range(len(DOCSTORE)):
                               if collection_name in DOCSTORE[i]:
                                    coll = DOCSTORE[i]
                                    break
                            with open(coll, 'r') as file:
                                data = json.load(file)["docstore/data"]

                            for i in ids_to_delete_existing:
                                del data[i]

                            final_dict = {}
                            final_dict["docstore/data"] = data


                            with open(coll, 'w') as file:
                                json.dump(final_dict, file, indent=4)

                            if ids_to_delete_existing:
                                collection.delete(ids=ids_to_delete_existing)

                                # Update session state after deletion
                                del st.session_state.doc_name_to_id[
                                    selected_doc_to_delete
                                ]
                                st.session_state.doc_list.remove(selected_doc_to_delete)

                                st.success(
                                    f"Document '{selected_doc_to_delete}' deleted successfully!"
                                )
                        except Exception as e:
                            st.error(f"Error deleting document: {str(e)}")
                    else:
                        st.warning(
                            f"No chunks found for document '{selected_doc_to_delete}'."
                        )
                else:
                    st.warning("**No document selected for deletion.**")

    with st.sidebar:
            st.markdown(
                    """
                <div>
                    <h1 style="text-align: center;">Evaluation Dashboard</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            for title, config in gauge_config.items():
                # Create the gauge chart
                fig = create_circular_gauge_chart(title, config["value"], 0, 100, config["color"], "")
                # Display the gauge chart
                st.plotly_chart(fig, use_container_width=True)
def user_page():

    if "selected_role" not in st.session_state:
        st.session_state.selected_role = ROLE[0]

    # Create a selectbox for role selection
    selected_role = st.selectbox("**Select your role**", ROLE)

    # Update the session state with the selected role
    st.session_state.selected_role = selected_role

    if st.session_state.selected_role == ROLE[0]:
        hr_team_page()
    elif st.session_state.selected_role == ROLE[1]:
        legal_team_page()
    elif st.session_state.selected_role == ROLE[2]:
        finance_team_page()
    elif st.session_state.selected_role == ROLE[3]:
        operations_team_page()
    elif st.session_state.selected_role == ROLE[4]:
        medical_team_page()
    elif st.session_state.selected_role == ROLE[5]:
        insurance_team_page()
    elif st.session_state.selected_role == ROLE[6]:
        LD_team_page()
    elif st.session_state.selected_role == ROLE[7]:
        others_team_page()


def hr_team_page():
    st.subheader(PAGE[1])
    query_page(COLLECTION[0], DATABASE[0], admin=False)


def legal_team_page():
    st.subheader(PAGE[2])
    query_page(COLLECTION[1], DATABASE[1], admin=False)


def finance_team_page():
    st.subheader(PAGE[3])
    query_page(COLLECTION[2], DATABASE[2], admin=False)


def operations_team_page():
    st.subheader(PAGE[4])
    query_page(COLLECTION[3], DATABASE[3], admin=False)


def medical_team_page():
    st.subheader(PAGE[5])
    query_page(COLLECTION[4], DATABASE[4], admin=False)


def insurance_team_page():
    st.subheader(PAGE[6])
    query_page(COLLECTION[5], DATABASE[5], admin=False)


def LD_team_page():
    st.subheader(PAGE[7])
    query_page(COLLECTION[6], DATABASE[6], admin=False)


def others_team_page():
    st.subheader(PAGE[8])
    query_page(COLLECTION[7], DATABASE[7], admin=False)


def show_documents(collection, key_prefix):
    # Initialize or update session state for doc_name_to_id and doc_list
    if f"{key_prefix}_doc_name_to_id" not in st.session_state:
        st.session_state[f"{key_prefix}_doc_name_to_id"] = {}
    if f"{key_prefix}_doc_list" not in st.session_state:
        docs = collection.get()["metadatas"]
        st.session_state[f"{key_prefix}_doc_name_to_id"] = {
            meta["source"].split("\\")[-1]: doc_id
            for doc_id, meta in zip(collection.get()["ids"], docs)
            if "source" in meta
        }
        st.session_state[f"{key_prefix}_doc_list"] = list(
            st.session_state[f"{key_prefix}_doc_name_to_id"].keys()
        )

    if st.button("Show Document", key=f"{key_prefix}_show_document"):
        # Refresh doc_name_to_id and doc_list in session state
        docs = collection.get()["metadatas"]
        st.session_state[f"{key_prefix}_doc_name_to_id"] = {
            meta["source"].split("\\")[-1]: doc_id
            for doc_id, meta in zip(collection.get()["ids"], docs)
            if "source" in meta
        }
        st.session_state[f"{key_prefix}_doc_list"] = list(
            st.session_state[f"{key_prefix}_doc_name_to_id"].keys()
        )
        st.selectbox(
            "Documents",
            st.session_state[f"{key_prefix}_doc_list"],
            key=f"{key_prefix}_doc_select",
            placeholder="See the document...",
        )


def pass_link(links):
    # Create a container for the links
    with st.container():
        with st.expander("URL of source dcuments"):
            for link in links:
                st.write(link)

    # Set the height of the container to make it scrollable
    st.markdown(
        """
            <style>
            .streamlit-expanderHeader {
                font-size: 20px;
                font-weight: bold;
            }
            .streamlit-expanderContent {
                max-height: 200px;
                overflow-y: auto;
            }
            </style>
            """,
        unsafe_allow_html=True,
    )


def get_links_by_collection(collection_name):
    if collection_name == "human_resources_documents":
        pass_link(human_resources_documents_link)

    elif collection_name == "legal_documents":
        pass_link(legal_documents_link)

    elif collection_name == "finance_documents":
        pass_link(finance_documents_link)

    elif collection_name == "operation_documents":
        pass_link(operation_documents_link)

    elif collection_name == "healthcare_documents":
        pass_link(healthcare_documents_link)

    elif collection_name == "insurance_documents":
        pass_link(insurance_documents_link)

    elif collection_name == "LD_documents":
        pass_link(LD_documents_link)

    elif collection_name == "others_documents":
        pass_link(others_documents_link)


def query_page(collection_name, db_path, admin):
    llm = Settings.llm
    embed_model = Settings.embed_model
    # Initialize Chroma collection
    collection = init_chroma_collection(db_path, collection_name)
    # Create columns for buttons
    col1, col2 = st.columns([10, 1])  # Adjust the ratio if needed


    if customer_self_demo_flag:
        get_links_by_collection(collection_name)
        select_database_table_question_csv = collection_name + ".csv"

        questions = pd.read_csv(select_database_table_question_csv)
        pre_defined_questions = st.selectbox(
            "Some frequently asked questions", questions
        )
        Frequently_asked_questions = None
        if st.button("Choose"):
            Frequently_asked_questions = pre_defined_questions

    if "temp_index" in st.session_state and st.session_state.temp_index:

        def callback():
            if st.session_state.STT_output:
                st.session_state["temp_question"] = st.session_state.STT_output

        temp_index = st.session_state.temp_index

        # Set up a prompt template for the question-answering task
        qa_prompt_str_temp = QA_PROMPT_STR

        # Set up a retriever to get relevant nodes
        retriever = temp_index.as_retriever(similarity_top_k=3)
        # Initialize chat history
        if "message" not in st.session_state:
            st.session_state.message = []

        # Store the previous response
        if "previous_response" not in st.session_state:
            st.session_state.previous_response = ""

        # Function to reset chat
        def reset_chat():
            st.session_state.message = []
            st.session_state.previous_response = ""

        # Place buttons in columns
        with col1:
            show_documents(collection, key_prefix="key_prefix")

        with col2:
            if st.button("Reset Chat", key="tq_reset"):
                reset_chat()

        # Option to select between text input or voice input
        input_option = st.radio(
            "**How would you like to ask your question?**",
            ("**Text**", "**Voice**"),
            key="kk",
        )

        if input_option == "**Text**":
            temp_questions = st.chat_input("Enter your question:")

        else:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Convert speech to text:**")
            with c2:
                speech_to_text(
                    language="en",
                    use_container_width=True,
                    just_once=True,
                    key="STT",
                    callback=callback,
                )
                temp_questions = st.session_state.get("temp_question", "")

        # Display chat messages from history on app rerun
        for mes in st.session_state.message:
            with st.chat_message(mes["role"]):
                st.markdown(mes["content"])

        if temp_questions:
            st.chat_message("user").markdown(temp_questions)
            st.session_state.message.append({"role": "user", "content": temp_questions})

            if temp_questions:
                # Retrieve nodes relevant to the question
                temp_retrieved_nodes = retriever.retrieve(temp_questions)
                context_str = "\n\n".join(
                    [
                        r.get_content().replace("{", "").replace("}", "")[:4000]
                        for r in temp_retrieved_nodes
                    ]
                )

                # Append the previous response to the context
                if (
                    st.session_state.previous_response
                    and "This question is outside the provided context."
                    not in st.session_state.previous_response
                ):
                    context_str = f"{context_str}\n\nPrevious Response: {st.session_state.previous_response}"
                # Format the QA prompt
                fmt_qa_prompt = qa_prompt_str_temp.format(
                    context_str=context_str, query_str=temp_questions
                )

                chat_text_qa_msgs = [
                    ChatMessage(role=MessageRole.SYSTEM, content=LLM_INSTRUCTION),
                    ChatMessage(role=MessageRole.USER, content=fmt_qa_prompt),
                ]
                text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

                # Query the index using the formatted prompt
                result = temp_index.as_query_engine(
                    text_qa_template=text_qa_template, llm=llm
                ).query(temp_questions)
                if result:
                    # Display the result
                    # formatted_response = textwrap.fill(result.response,width=170)  # Adjust the width as needed
                    formatted_response = result.response

                    with st.chat_message("assistant"):
                        st.code(f"{formatted_response}", language="None")
                    st.session_state.message.append(
                        {"role": "assistant", "content": f"{result.response}"}
                    )
                    st.session_state.previous_response = (
                        result.response
                    )  # Save the current response for future use

                    temp_questions = ""
                else:
                    st.warning(N_DOC)

    else:

        def callback():
            if st.session_state.STT_output:
                st.session_state["temp_question"] = st.session_state.STT_output

        # Initialize chat history
        if "message" not in st.session_state:
            st.session_state.message = []
        # Store the previous response
        if "previous_response" not in st.session_state:
            st.session_state.previous_response = ""
        if "question" not in st.session_state:
            st.session_state.question=""
        if "chunk" not in st.session_state:
            st.session_state.chunk=""
        if "source" not in st.session_state:
            st.session_state.source=""
        if "feedback_text" not in st.session_state:
            st.session_state.feedback_text={}

        # Function to reset chat
        def reset_chat():
            st.session_state.message = []
            st.session_state.previous_response = ""

        # Place buttons in columns
        with col1:
            show_documents(collection, key_prefix="key_prefix")

        with col2:
            if st.button("Reset Chat", key="q_reset"):
                reset_chat()
        # Option to select between text input or voice input
        input_option = st.radio(
            "**How would you like to ask your question?**",
            ("**Text**", "**Voice**"),
            key="kk",
        )


        if input_option == "**Text**":
            query = st.chat_input("Enter your question:")
            if customer_self_demo_flag:
                if query:
                    single_question = query
                else:
                    single_question = Frequently_asked_questions  # Use the selected FAQ
            else:
                single_question = query

        else:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Convert speech to text:**")
            with c2:
                speech_to_text(
                    language="en",
                    use_container_width=True,
                    just_once=True,
                    key="STT",
                    callback=callback,
                )
                single_question = st.session_state.get("temp_question", "")

        # Display chat messages from history on app rerun
        for mes in st.session_state.message:
            with st.chat_message(mes["role"]):
                st.markdown(mes["content"])

        if single_question:
            st.chat_message("user").markdown(single_question)
            st.session_state.message.append(
                {"role": "user", "content": single_question}
            )
            st.session_state.question=single_question
            with st.spinner("Generating response..."):

                # Process the single question if provided
                if (
                    "documents" in collection.get()
                    and len(collection.get()["documents"]) > 0
                ):
                    vector_store = ChromaVectorStore(chroma_collection=collection)
                    # docstore = SimpleDocumentStore.from_persist_path(f"./docstore_{collection_name}.json")
                    for i in range(len(DOCSTORE)):
                        if collection_name in DOCSTORE[i]:
                            coll = DOCSTORE[i]
                            break
                    docstore = SimpleDocumentStore.from_persist_path(coll)
                    storage_context = StorageContext.from_defaults(
                        docstore=docstore, vector_store=vector_store
                    )
                    vector_index = VectorStoreIndex(
                        nodes=[],
                        storage_context=storage_context,
                        embed_model=embed_model,
                    )

                    # Create the BM25 retriever
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore=docstore, similarity_top_k=int(BM25_TOP)
                    )

                    # Function to perform hybrid retrieval
                    def hybrid_retrieve(query, alpha=0.5):
                        # Get results from BM25
                        bm25_results = bm25_retriever.retrieve(query)

                        # Get results from the vector store
                        vector_results = vector_index.as_retriever(
                            similarity_top_k=int(VEC_TOP)
                        ).retrieve(query)

                        # Combine results with weighting
                        combined_results = {}
                        # Weight BM25 results
                        for result in bm25_results:
                            combined_results[result.id_] = combined_results.get(
                                result.id_, 0
                            ) + (1 - alpha)

                        # Weight vector results
                        for result in vector_results:
                            combined_results[result.id_] = (
                                combined_results.get(result.id_, 0) + alpha
                            )
                        # Sort results based on the combined score
                        sorted_results = sorted(
                            combined_results.items(), key=lambda x: x[1], reverse=True
                        )
                        # Return the top N results
                        return [
                            docstore.get_document(doc_id)
                            for doc_id, _ in sorted_results
                        ]


                    qa_prompt_str = QA_PROMPT_STR
                    alpha = float(
                        Weight_of_importance_for_retriver
                    )  # Adjust alpha as needed
                    retrieved_nodes = hybrid_retrieve(single_question, alpha)
                    ids = [doc.id_ for doc in retrieved_nodes]
                    # print(ids)
                    context_str = "\n\n".join(
                        [
                            r.get_content().replace("{", "").replace("}", "")
                            for r in retrieved_nodes
                        ]
                    )

                    # Append the previous response to the context
                    if (
                        st.session_state.previous_response
                        and "This question is outside the provided context."
                        not in st.session_state.previous_response
                    ):
                        context_str = (
                            f"{context_str}\n\n{st.session_state.previous_response}"
                        )

                    st.session_state.chunk=context_str
                    fmt_qa_prompt = qa_prompt_str.format(
                        context_str=context_str, query_str=single_question
                    )
                    chat_text_qa_msgs = [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                             content=LLM_INSTRUCTION,
                        ),
                        ChatMessage(role=MessageRole.USER, content=fmt_qa_prompt),
                    ]

                    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

                    query_engine = vector_index.as_query_engine(
                        text_qa_template=text_qa_template,
                        llm=llm,
                    )

                    response = query_engine.query(single_question)
                    # Initialize source variable
                    source = " and ".join(ids)  # Default to joining IDs
                    st.session_state.source=source
                    # Check if any instruction is present in the response
                    for instruction in METADATA_INSTRUCTION:
                        if instruction in str(response.response).lower():
                            source =NO_METADATA
                            st.session_state.source=source
                            break  # Exit the loop if a match is found

                    # print(source)  # Output the final value of source





                    formatted_response = response.response
                    # Display the response for the single question
                    with st.chat_message("assistant"):
                        if show_chunk:
                            st.code(
                                f"{formatted_response} --- Source : {source} --- Chunks :{context_str} ", language="None"
                            )
                        else:
                            st.code(
                                f"{formatted_response} --- Source : {source}", language="None"
                            )


                        append_to_csv(
                            collection_name,
                            single_question,
                            context_str,
                            str(response.response),
                            str(source),
                        )

                        st.session_state.message.append(
                            {
                                "role": "assistant",
                                "content": f"{response.response} --- Source: {source}",
                            }
                        )
                        response.response = (
                            response.response.replace("{", "")
                            .replace("}", "")
                            .replace("TP", "")
                        )
                        st.session_state.previous_response = (
                            response.response
                        )  # Save the current response for future use

                        # append_to_csv(collection_name, single_question, context_str, str(response.response), str(source))
                        # single_question = ""


                else:
                    st.warning(N_DOC)
        if st.session_state.previous_response:
            ques=st.session_state.question
            chunk=st.session_state.chunk
            cur_response=st.session_state.previous_response
            metadata=st.session_state.source
            feedback_text = st.text_input(f"*Provide feedback here*", key="feedback")
            voting_interface(collection_name, ques,chunk,cur_response,metadata, feedback_text)

def init_chroma_collection(db_path, collection_name):
    db = chromadb.PersistentClient(path=db_path)
    return db.get_or_create_collection(collection_name, embedding_function=openai_ef)


def use_llamaparse(file_content, file_name):
    with open(file_name, "wb") as f:
        f.write(file_content)

    parser = LlamaParse(result_type="text", verbose=True, language="en", num_workers=2)
    documents = parser.load_data([file_name])

    os.remove(file_name)

    res = ""
    for i in documents:
        res += i.text + " "
    return res


def upload_to_blob_storage(
    connect_str: str, container_name: str,collection_name, file_content, file_name
):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Ensure the container exists and create if necessary
    container_client = blob_service_client.get_container_client(container_name)
    blob_name = f"{collection_name}/{file_name}"
    blob_client = container_client.get_blob_client(blob_name)

    print(f"Uploading {file_name} to {blob_name}...")
    blob_client.upload_blob(file_content, overwrite=True)
    # # Upload the file to Azure Blob Storage
    # blob_client = container_client.get_blob_client(file_name)
    # print(f"Uploading {file_name} to blob storage...")
    # blob_client.upload_blob(file_content, overwrite=True)


def delete_from_blob_storage(connect_str: str, container_name: str, file_name: str,collection_name):
    # Create a BlobServiceClient to interact with the Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

     # Get the container client
    container_client = blob_service_client.get_container_client(container_name)

    # Create the blob name with the collection prefix
    blob_name = f"{collection_name}/{file_name}"

    # Get the blob client for the specific file (blob)
    blob_client = container_client.get_blob_client(blob_name)

    # Delete the specified blob (file)
    try:
        print(f"Deleting {blob_name} from blob storage...")
        blob_client.delete_blob()
        print(
            f"Blob '{blob_name}' deleted successfully from container '{container_name}'."
        )
    except Exception as e:
        print(f"Failed to delete blob: {e}")

    # # Get the container client
    # container_client = blob_service_client.get_container_client(container_name)

    # # Get the blob client for the specific file (blob)
    # blob_client = container_client.get_blob_client(file_name)

    # # Delete the specified blob (file)
    # try:
    #     print(f"Deleting {file_name} from blob storage...")
    #     blob_client.delete_blob()
    #     print(
    #         f"Blob '{file_name}' deleted successfully from container '{container_name}'."
    #     )
    # except Exception as e:
    #     print(f"Failed to delete blob: {e}")


def use_unstructured(uploaded_file_path, file_name):

    # Configure the pipeline
    pipeline = Pipeline.from_configs(
        context=ProcessorConfig(),
        indexer_config=LocalIndexerConfig(
            input_path=uploaded_file_path
        ),  # Use the uploaded file path
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key=unstructured_api_key,
            partition_endpoint=unstructured_api_url,
            strategy="hi_res",
        ),
        uploader_config=LocalUploaderConfig(
            output_dir=OUTPUT_FOLDER_FOR_UNSTRUCTURED
        ),  # Specify the output directory
    )

    # Run the pipeline
    pipeline.run()

    # Find all JSON output files in the "output" folder
    output_dir = OUTPUT_FOLDER_FOR_UNSTRUCTURED
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    concatenated_text = ""

    if json_files:
        # Loop through each JSON file and extract its contents
        for json_file in json_files:
            output_file_path = os.path.join(output_dir, json_file)
            with open(output_file_path, "r") as f:
                output_data = json.load(f)

            # Extract text for each element_id
            for item in output_data:  # Assuming output_data is a list of dictionaries
                text = item.get("text", "")

                # Concatenate the text if it's not empty
                if text:
                    concatenated_text += text + " "  # Add a space between texts

    else:
        return "No JSON output found in the 'output' folder."
    os.remove(uploaded_file_path)
    # Remove the output folder after processing
    if os.path.exists(OUTPUT_FOLDER_FOR_UNSTRUCTURED):
        shutil.rmtree(
            OUTPUT_FOLDER_FOR_UNSTRUCTURED
        )  # This will delete the output folder and all its contents

    return (
        concatenated_text.strip()
    )  # Return the concatenated text without trailing spaces
def voting_interface(collection_name, single_question, context_str, response, source,feedback):
    votes = load_votes(collection_name)
    col1, col2= st.columns(2)
    with col1:
        if st.button(f"👍{votes['upvotes']}"):
            votes["upvotes"] += 1
            print(votes["upvotes"])
            save_votes(collection_name, votes)
            insert_feedback(collection_name, single_question, context_str, response, source,"like",feedback)
            st.rerun()
            return 1
    with col2:
        if st.button(f"👎 {votes['downvotes']}"):
            votes["downvotes"] += 1
            print(votes["downvotes"])
            save_votes(collection_name, votes)
            insert_feedback(collection_name, single_question, context_str, response, source, "dislike",feedback)
            st.rerun()
            return 1

if __name__ == "__main__":
    main()
