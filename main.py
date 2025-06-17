# main.py
import streamlit as st
import pymupdf
from sentence_transformers import SentenceTransformer
import datetime
import time
import chromadb
import os
import requests
import json
from PIL import Image
import base64
import io

# --- IMPORTS FOR LOCAL LLM ---
import ollama
# ---------------------------------

# --- CONFIGURATION FOR GOOGLE FORMS LOGGING ---
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfdtpLXu7Q8YZQVQVRR_d-txzpqGdXIQQmhQ4nzbt943Hlf5-Q/formResponse"
FORM_ENTRY_IDS = {
    "user_name": "entry.709570487",
    "log_date": "entry.206826878",
    "log_time": "entry.1952110840",
    "event_type": "entry.2096410659",
    "query_1": "entry.2075974249",
    "query_2": "entry.1640396065",
    "query_3": "entry.857834386",
    "query_4": "entry.1182424791",
    "query_5": "entry.452996460",
    "query_6": "entry.2134939414",
    "query_7": "entry.2044152134",
    "query_8": "entry.1438153575",
    "query_9": "entry.1343282068",
    "query_10": "entry.698785579"
}
# ---------------------------------------------

# --- PASSWORD FOR UPLOADS ---
PASSWORD_FOR_UPLOAD = "1234"

# --- PRE-LOADED DOCUMENTS CONFIGURATION ---
PRELOAD_PDF_DIR = "preloaded_documents"
PRELOAD_BATCH_ID = "initial_30_docs_v1"

# --- TRIAL PERIOD CONFIGURATION ---
TRIAL_DAYS = 30
LICENSE_FILE = "license.json"
INSTALL_MARKER_FILE = ".installed_marker" # New: Marker for successful installation
PROGRAM_INSTALL_ID = "program_installation_date" # New: ID for program-wide install date in license.json
# -----------------------------------------------

# --- LOCAL LLM CONFIGURATION ---
LOCAL_LLM_MODEL = "phi3"

# Logging Function (Keep as is)
def log_activity_to_google_form(user_name, event_type, query_content=""):
    current_datetime = datetime.datetime.now()
    log_date = current_datetime.strftime("%Y-%m-%d")
    log_time = current_datetime.strftime("%H:%M:%S")

    if "current_client_log_data" not in st.session_state:
        st.session_state.current_client_log_data = {
            "user_name": user_name,
            "log_date": log_date,
            "log_time": log_time,
            "queries": [""] * 10,
            "query_count": 0
        }
        if event_type == "New Session":
            form_data = {
                FORM_ENTRY_IDS["user_name"]: user_name,
                FORM_ENTRY_IDS["log_date"]: log_date,
                FORM_ENTRY_IDS["log_time"]: log_time,
                FORM_ENTRY_IDS["event_type"]: event_type,
                FORM_ENTRY_IDS["query_1"]: "",
                FORM_ENTRY_IDS["query_2"]: "",
                FORM_ENTRY_IDS["query_3"]: "",
                FORM_ENTRY_IDS["query_4"]: "",
                FORM_ENTRY_IDS["query_5"]: "",
                FORM_ENTRY_IDS["query_6"]: "",
                FORM_ENTRY_IDS["query_7"]: "",
                FORM_ENTRY_IDS["query_8"]: "",
                FORM_ENTRY_IDS["query_9"]: "",
                FORM_ENTRY_IDS["query_10"]: ""
            }
            try:
                requests.post(GOOGLE_FORM_URL, data=form_data)
            except requests.exceptions.RequestException as e:
                st.error(f"Network error logging new session: {e}")
            except Exception as e:
                st.error(f"Error logging new session: {e}")
            return

    if event_type == "Chat Query" and st.session_state.current_client_log_data["query_count"] < 10:
        query_index = st.session_state.current_client_log_data["query_count"]
        st.session_state.current_client_log_data["queries"][query_index] = query_content
        st.session_state.current_client_log_data["query_count"] += 1

        form_data = {
            FORM_ENTRY_IDS["user_name"]: user_name,
            FORM_ENTRY_IDS["log_date"]: log_date,
            FORM_ENTRY_IDS["log_time"]: log_time,
            FORM_ENTRY_IDS["event_type"]: event_type,
        }
        query_key = f"query_{st.session_state.current_client_log_data['query_count']}"
        if query_key in FORM_ENTRY_IDS:
            form_data.update({FORM_ENTRY_IDS.get(query_key, ""): query_content})
        else:
            form_data.update({FORM_ENTRY_IDS.get("event_type"): f"Query {st.session_state.current_client_log_data['query_count']} (Exceeded 10 tracked queries)"})
            form_data.update({FORM_ENTRY_IDS.get("query_1"): query_content})

        try:
            requests.post(GOOGLE_FORM_URL, data=form_data)
        except requests.exceptions.RequestException as e:
            st.error(f"Network error logging query: {e}")
        except Exception as e:
            st.error(f"Error logging query: {e}")

    elif event_type != "New Session":
        form_data = {
            FORM_ENTRY_IDS["user_name"]: user_name,
            FORM_ENTRY_IDS["log_date"]: log_date,
            FORM_ENTRY_IDS["log_time"]: log_time,
            FORM_ENTRY_IDS["event_type"]: event_type,
            FORM_ENTRY_IDS["query_1"]: query_content,
        }
        try:
            requests.post(GOOGLE_FORM_URL, data=form_data)
        except requests.exceptions.RequestException as e:
            st.error(f"Network error logging event: {e}")
        except Exception as e:
            st.error(f"Error logging event: {e}")

# Document processing (Keep as is)
def extract_text_from_pdf(pdf_file):
    try:
        if isinstance(pdf_file, str):
            doc = pymupdf.open(pdf_file)
            file_name = os.path.basename(pdf_file)
        else:
            doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
            file_name = pdf_file.name

        all_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                all_text += f"\n\n=== Page {page_num + 1} from {file_name} ===\n{text}"
        doc.close()
        return all_text
    except Exception as e:
        if isinstance(pdf_file, str):
            st.error(f"Error reading {os.path.basename(pdf_file)}: {str(e)}")
        else:
            st.error(f"Error reading {pdf_file.name}: {str(e)}")
        return ""

# Image processing (Keep as is)
def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_image_with_huggingface(image, question):
    try:
        img_base64 = encode_image_to_base64(image)
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        response = requests.post(
            API_URL,
            headers={"Authorization": "Bearer hf_demo"},
            json={"inputs": img_base64}
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get('generated_text', 'Unable to analyze image')
                return f"Image Analysis (Hugging Face): {caption}\n\nBased on your question '{question}', this image appears to show: {caption}"
            else:
                return "I can see the image but couldn't generate a detailed description."
        else:
            return f"Image analysis service (Hugging Face) temporarily unavailable (Status: {response.status_code})."
    except Exception as e:
        return f"Error analyzing image with Hugging Face: {str(e)}."

def analyze_image_with_local_description(image, question):
    try:
        width, height = image.size
        format_info = image.format or "Unknown"
        mode = image.mode
        analysis = f"""Image Analysis Results (Local):

**Image Properties:**
- Dimensions: {width} x {height} pixels
- Format: {format_info}
- Color Mode: {mode}

**Your Question:** {question}

**Basic Analysis:** I can see this is a {width}x{height} pixel {format_info.lower()} image. To provide more detailed analysis about the content, you might want to:

1.  **Describe what you see** in your question for better context
2.  **Ask specific questions** about elements in the image
3.  **Upload related documents** that might provide context

**Tip:** For best results, try questions like:
- "What text can you see in this image?"
- "Describe the main elements in this picture"
- "What type of document or diagram is this?"
"""
        return analysis
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Load simple models (cached for speed)
@st.cache_resource
def setup_simple_search():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path="./simple_db")
        collection = client.get_or_create_collection("docs")
        return embedder, collection
    except Exception as e:
        st.error(f"Error setting up search: {str(e)}")
        return None, None

# Simple search function (Keep as is)
def search_documents(question):
    embedder, collection = setup_simple_search()
    if not embedder or not collection:
        return []
    try:
        results = collection.query(
            query_texts=[question],
            n_results=5
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# Split text into chunks (Keep as is)
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if end >= len(text):
            break
    return chunks

# Process pre-loaded documents (Keep as is)
@st.cache_resource(show_spinner=False)
def process_preloaded_documents():
    embedder, collection = setup_simple_search()
    if not embedder or not collection:
        st.error("Failed to set up document processing for pre-loaded files.")
        return

    try:
        test_query = collection.get(ids=[PRELOAD_BATCH_ID])
        if test_query['ids'] and test_query['ids'][0] == PRELOAD_BATCH_ID:
            st.info(f"Pre-loaded documents ({PRELOAD_BATCH_ID}) already processed.")
            return
    except Exception as e:
        pass

    st.warning("First-time setup: Processing pre-loaded documents. This may take a moment...")
    placeholder = st.empty()

    if not os.path.exists(PRELOAD_PDF_DIR):
        placeholder.error(f"Error: Pre-loaded PDF directory '{PRELOAD_PDF_DIR}' not found. Please ensure it exists and contains PDFs.")
        return

    pdf_files = [f for f in os.listdir(PRELOAD_PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        placeholder.warning(f"No PDF files found in '{PRELOAD_PDF_DIR}'.")
        return

    total_files_to_process = len(pdf_files)
    processed_count = 0
    total_chunks_processed = 0

    for idx, filename in enumerate(pdf_files):
        file_path = os.path.join(PRELOAD_PDF_DIR, filename)
        placeholder.text(f"Processing pre-loaded: {filename} ({idx + 1}/{total_files_to_process})")

        text = extract_text_from_pdf(file_path)

        if text.strip():
            chunks = split_text_into_chunks(text)
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    doc_id = f"preloaded_{filename.replace('.', '_')}_chunk_{chunk_idx}"
                    collection.add(
                        documents=[chunk],
                        ids=[doc_id]
                    )
                    total_chunks_processed += 1
                except Exception as e:
                    placeholder.warning(f"Error processing chunk from pre-loaded {filename}: {str(e)}")
        processed_count += 1
    
    try:
        collection.add(
            documents=["This document serves as a marker for processed pre-loaded batch."],
            ids=[PRELOAD_BATCH_ID]
        )
    except Exception as e:
        placeholder.error(f"Could not add batch marker to ChromaDB: {e}")

    placeholder.success(f"✅ Pre-loaded documents processed. Total {total_files_to_process} files, {total_chunks_processed} chunks.")
    time.sleep(3)
    placeholder.empty()

# Generate answer with local LLM (Keep as is)
def generate_answer_with_ollama(question, context_docs, user_name):
    if not context_docs:
        return f"Sorry {user_name}, I couldn't find enough relevant information in your uploaded documents to answer that question. Please try rephrasing or uploading more related content."

    context = "\n\n".join(context_docs)
    prompt_template = f"""
    You are an AI assistant specialized in providing information from provided documents.
    Answer the following question based ONLY on the provided context. If the answer cannot be found
    in the context, politely state that you don't have enough information (e.g., "I cannot find that information in the provided documents.").
    Keep your answer concise, factual, and directly related to the question.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    try:
        response = ollama.chat(
            model=LOCAL_LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt_template}],
            options={'temperature': 0.1, 'num_predict': 256, 'seed': 42},
            stream=False
        )
        return response['message']['content']
    except requests.exceptions.ConnectionError:
        return f"Could not connect to Ollama. Please ensure Ollama is running and the model '{LOCAL_LLM_MODEL}' is downloaded. Try running 'ollama run {LOCAL_LLM_MODEL}' in your terminal."
    except Exception as e:
        return f"Error communicating with the local LLM ({LOCAL_LLM_MODEL}). Error: {str(e)}. Please check your Ollama installation and model."

# Function to handle chat responses (Keep as is)
def handle_chat_response(prompt):
    if "user_name" not in st.session_state or st.session_state.user_name is None:
        st.warning("Please tell me your name first!")
        return

    # --- NEW: Check GLOBAL trial status for all features ---
    program_status, _, _ = check_program_trial_status()
    if program_status == "expired":
        st.error(f"The program's trial period has ended. Please contact support to purchase the full version and continue using the AI chat.")
        return
    # --------------------------------------------------------

    log_activity_to_google_form(st.session_state.user_name, "Chat Query", prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching documents and asking {LOCAL_LLM_MODEL} to generate response..."):
            relevant_docs = search_documents(prompt)

            if relevant_docs:
                llm_response = generate_answer_with_ollama(prompt, relevant_docs, st.session_state.user_name)

                sources_info = "\n\n**Sources Used:**\n"
                for i, doc in enumerate(relevant_docs[:3]):
                    lines = doc.split('\n')
                    source_line = next((line for line in lines if '=== Page' in line and 'from' in line), None)
                    if source_line:
                        sources_info += f"- {source_line.replace('===', '').strip()}\n"
                    else:
                        sources_info += f"- Document chunk {i+1}\n"

                final_response = f"{llm_response}\n\n{sources_info}"
                st.write(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            else:
                no_results_msg = (
                    f"Sorry {st.session_state.user_name}, I couldn't find relevant information in your uploaded documents "
                    f"to answer that. This might be because:\n\n"
                    f"- The information isn't present in the documents you've uploaded.\n"
                    f"- The keywords you used didn't match the document content well enough.\n"
                    f"- Documents haven't been processed yet. Please upload and click 'Process Documents'."
                    f"\n\nPlease try rephrasing your question or uploading more related content. "
                    f"If your question is about an image, please use the 'Image Analysis' section."
                )
                st.write(no_results_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_results_msg})

# --- NEW/UPDATED: TRIAL MANAGEMENT FUNCTIONS ---
def load_license_data():
    if os.path.exists(LICENSE_FILE):
        with open(LICENSE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, return empty and try to recreate
                print(f"Warning: {LICENSE_FILE} is corrupted. Recreating.")
                return {}
    return {}

def save_license_data(data):
    with open(LICENSE_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def check_program_trial_status():
    """Checks the global program trial status based on the INSTALL_MARKER_FILE
       and the installation date in license.json."""
    
    license_data = load_license_data()
    install_info = license_data.get(PROGRAM_INSTALL_ID)

    current_date = datetime.date.today()

    if not os.path.exists(INSTALL_MARKER_FILE) or install_info is None:
        # Program not properly installed or marker/date missing
        return "uninstalled", None, None # Status, Start Date, Days Left
    
    install_date_str = install_info.get("install_date")
    if install_date_str is None: # Should not happen if install_info exists
        return "uninstalled", None, None

    install_date = datetime.datetime.strptime(install_date_str, "%Y-%m-%d").date()
    days_passed = (current_date - install_date).days
    days_left = TRIAL_DAYS - days_passed

    if days_left <= 0:
        return "expired", install_date, 0
    else:
        return "active", install_date, days_left


def check_user_trial_status(username):
    """Checks individual user trial status against the global program trial."""
    
    program_status, _, _ = check_program_trial_status()
    if program_status == "uninstalled":
        # Cannot determine user trial if program isn't installed
        return "uninstalled", 0
    elif program_status == "expired":
        return "expired", 0

    # Program is active, check user's trial within this active period
    license_data = load_license_data()
    user_data = license_data.get(username)

    current_date = datetime.date.today()

    if user_data is None:
        # First time this user logs in, start their trial
        trial_start_date_str = current_date.strftime("%Y-%m-%d")
        license_data[username] = {
            "trial_start_date": trial_start_date_str,
            "trial_status": "active"
        }
        save_license_data(license_data)
        return "active", TRIAL_DAYS # Initial days left

    trial_start_date = datetime.datetime.strptime(user_data["trial_start_date"], "%Y-%m-%d").date()
    days_passed = (current_date - trial_start_date).days
    days_left = TRIAL_DAYS - days_passed

    if days_left <= 0:
        user_data["trial_status"] = "expired"
        save_license_data(license_data)
        return "expired", 0
    else:
        user_data["trial_status"] = "active"
        save_license_data(license_data)
        return "active", days_left

def display_trial_info(user_name):
    # First, check program status
    program_status, install_date, prog_days_left = check_program_trial_status()

    if program_status == "uninstalled":
        st.error("🚨 Program not installed. Please run `setup.py` and `install_ollama_and_model.bat`.")
        st.stop() # Stop Streamlit execution here
    elif program_status == "expired":
        st.error("🚨 The program's overall trial period has ended. Please contact sales to purchase the full version.")
        return # Don't proceed with user-specific trial info

    # If program is active, then check user's individual trial
    user_trial_status, user_days_left = check_user_trial_status(user_name)

    if user_trial_status == "active":
        if user_days_left <= 7:
            st.warning(f"🔔 Your trial will expire in **{user_days_left} day(s)**. Please contact sales to purchase the full version!")
        else:
            st.info(f"✨ You are currently on a **{TRIAL_DAYS}-day free trial**. Days remaining: **{user_days_left}**.")
    else: # user_trial_status == "expired"
        st.error("🚨 Your individual trial period has ended. Most features are disabled. Please contact sales to purchase the full version.")
# ----------------------------------------------------

# Main app
def main():
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # --- NEW: Initial Program Installation Check ---
    program_status, _, _ = check_program_trial_status()
    if program_status == "uninstalled":
        st.error("--- PROGRAM NOT INSTALLED ---")
        st.markdown("It seems the AI Document Assistant is not properly installed.")
        st.markdown("Please run the setup scripts in the following order:")
        st.markdown("1. Double-click `setup.py`")
        st.markdown("2. Double-click `install_ollama_and_model.bat`")
        st.markdown("Refer to the installation guide for detailed steps.")
        st.stop() # Stop Streamlit from rendering the rest of the app
    # -----------------------------------------------

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #870F26;
            color: #FFFFFF;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
        }
        p {
            color: #FFFFFF;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        div[data-testid="stChatMessage"] p {
            color: #FFFFFF;
        }
        .stTextInput label, .stFileUploader label {
            color: #FFFFFF;
        }
        .stTextInput > div > div > input {
            border: 3px solid #870F26;
        }
        div[data-testid="stChatInput"] div[data-testid="stTextInput"] div[data-testid="stForm"] > div > div > input {
           border: 3px solid #870F26 !important;
        }
        div[data-testid="stChatInput"] {
            border: 3px solid #870F26;
            border-radius: 0.5rem;
            padding: 0.5rem;
        }
        div[data-testid="stAlert"] {
            color: unset !important;
            background-color: unset !important;
        }
        div[data-testid="stAlert"] p {
             color: unset !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 AI Document Assistant")
    st.write("Explore your documents with AI assistance!")

    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    if st.session_state.user_name is None:
        user_name_input = st.text_input("Hey there, how shall I address you?", key="initial_name_input")
        if user_name_input:
            st.session_state.user_name = user_name_input.strip()
            # --- NEW: Initialize user trial ---
            check_user_trial_status(st.session_state.user_name)
            # -------------------------------------------
            log_activity_to_google_form(st.session_state.user_name, "New Session")
            st.session_state.messages = []
            st.success(f"Nice to meet you, {st.session_state.user_name}! You can now ask questions or upload documents.")
            time.sleep(1)
            st.rerun()
        return
    else:
        st.markdown(f"<p style='font-size: 20px;'>Welcome back, <span style='font-size: 24px; font-weight: bold;'>{st.session_state.user_name}</span>! How can I help you today?</p>", unsafe_allow_html=True)
        # --- NEW: Display trial info for the user ---
        display_trial_info(st.session_state.user_name)
        # --------------------------------------------

    col1, col2, col3 = st.columns([1, 2, 1])

    # --- NEW: Get program and user trial status for conditional rendering ---
    program_status, _, _ = check_program_trial_status()
    user_trial_status, _ = check_user_trial_status(st.session_state.user_name)
    is_program_active = (program_status == "active")
    is_user_trial_active = (user_trial_status == "active")
    # Features are enabled only if *both* program and user trial are active
    can_use_features = is_program_active and is_user_trial_active
    # -----------------------------------------------------------------------


    with col1:
        st.header("📁 Document Management")

        if "preloaded_docs_processed" not in st.session_state:
            st.session_state.preloaded_docs_processed = False

        if not st.session_state.preloaded_docs_processed:
            process_preloaded_documents()
            st.session_state.preloaded_docs_processed = True

        st.info(f"✨ **You currently have access to our pre-loaded documents.**")

        if can_use_features: # Only show password prompt if program and user trial are active
            st.markdown(f"To upload your own documents, please enter the password below.")
            password_entered = st.text_input("Enter password to enable custom document upload:", type="password", key="upload_password")

            if password_entered == PASSWORD_FOR_UPLOAD:
                st.session_state.password_correct = True
                st.success("Password accepted! You can now upload your own documents.")
            elif password_entered and password_entered != PASSWORD_FOR_UPLOAD:
                st.session_state.password_correct = False
                st.error("Incorrect password. Please try again.")
            elif "password_correct" not in st.session_state:
                st.session_state.password_correct = False

            if st.session_state.password_correct:
                uploaded_files = st.file_uploader(
                    "Choose PDF files to add to your knowledge base",
                    type=['pdf'],
                    accept_multiple_files=True,
                    help="Upload multiple PDF files to create your knowledge base"
                )
                if st.button("🔄 Process My Documents", type="primary", use_container_width=True):
                    if uploaded_files:
                        embedder, collection = setup_simple_search()
                        if embedder and collection:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            total_files = len(uploaded_files)
                            processed_chunks = 0
                            for file_idx, pdf_file in enumerate(uploaded_files):
                                status_text.text(f"Processing {pdf_file.name}...")
                                text = extract_text_from_pdf(pdf_file)
                                if text.strip():
                                    chunks = split_text_into_chunks(text)
                                    for chunk_idx, chunk in enumerate(chunks):
                                        try:
                                            doc_id = f"custom_{pdf_file.name.replace('.', '_')}_chunk_{file_idx}_{chunk_idx}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                                            collection.add(
                                                documents=[chunk],
                                                ids=[doc_id]
                                            )
                                            processed_chunks += 1
                                        except Exception as e:
                                            st.warning(f"Error processing chunk from {pdf_file.name}: {str(e)}")
                                    progress_bar.progress((file_idx + 1) / total_files)
                            status_text.text("✅ Custom document processing complete!")
                            st.success(f"Successfully added {total_files} of your documents into {processed_chunks} searchable chunks!")
                            time.sleep(2)
                            progress_bar.empty()
                            status_text.empty()
                    else:
                        st.warning("Please upload at least one PDF file to process.")
                st.info("💡 **Tips for Custom Uploads:**\n- Upload multiple PDFs at once\n- Supported formats: PDF only\n- Processing may take a few minutes for large files")
            else:
                 st.info("🔒 Enter the password above to unlock the ability to upload your own documents.")
        else: # Program or user trial expired
            st.warning("🔒 Uploading new documents is disabled. Your program trial or individual user trial has expired. Please purchase the full version.")


    with col2:
        st.header("💬 Ask Questions")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message.get("image"):
                        st.image(message["image"], caption="Uploaded Image", use_column_width=True)
                    st.write(message["content"])

        st.write("---")
        st.subheader("Quick Queries")

        if can_use_features:
            if st.button("Give me a Summary of a key document", use_container_width=True):
                handle_chat_response("Give me a summary of a key document about construction project management from the pre-loaded files.")

            if st.button("What are common safety regulations?", use_container_width=True):
                handle_chat_response("What are common safety regulations mentioned in the documents?")

            if st.button("Explain a technical concept from the documents", use_container_width=True):
                handle_chat_response("Explain a technical concept like 'grounding resistance' based on the pre-loaded documents.")
        else:
            st.info("Quick queries are disabled because the program trial or your individual trial has expired.")

        st.write("---")

        if can_use_features:
            if prompt := st.chat_input("Ask a question about the documents or images..."):
                handle_chat_response(prompt)
        else:
            st.text_input("Ask a question about the documents or images...", disabled=True, placeholder="Program trial or your individual trial has expired. Chat is disabled.")

        if st.button("🗑️ Clear Chat History"):
            log_activity_to_google_form(st.session_state.user_name, "Chat History Cleared")
            st.session_state.messages = []
            st.session_state.user_name = None
            st.session_state.current_client_log_data = None
            st.rerun()

    with col3:
        st.header("🖼️ Image Analysis")
        if can_use_features:
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload images for AI analysis"
            )
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                image_question = st.text_input(
                    "Ask about this image:",
                    placeholder="What do you see in this image?",
                    help="Ask specific questions about the image content"
                )
                if st.button("🔍 Analyze Image", use_container_width=True):
                    if image_question:
                        with st.spinner("Analyzing image..."):
                            analysis = analyze_image_with_huggingface(image, image_question)
                            if "temporarily unavailable" in analysis or "Error analyzing image with Hugging Face" in analysis:
                                 analysis = analyze_image_with_local_description(image, image_question)

                            log_activity_to_google_form(st.session_state.user_name, "Image Analysis", image_question)
                            st.session_state.messages.append({
                                "role": "user",
                                "content": f"[Image uploaded] {image_question}",
                                "image": image
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": analysis
                            })
                            st.rerun()
                    else:
                        st.warning("Please ask a question about the image.")
            st.info("🖼️ **Image Tips:**\n- Supports: PNG, JPG, JPEG, GIF, BMP\n- Ask specific questions\n- Works with diagrams, charts, photos\n- Combines with document search")
        else:
            st.warning("Image analysis is disabled because the program trial or your individual trial has expired.")
            st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'gif', 'bmp'], disabled=True)
            st.text_input("Ask about this image:", disabled=True, placeholder="Trial expired.")
            st.button("🔍 Analyze Image", disabled=True)

    st.markdown("---")
    st.markdown("Copyright by Werner Ton 2025", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
