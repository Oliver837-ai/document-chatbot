import streamlit as st
import pymupdf  # for PDFs
from sentence_transformers import SentenceTransformer
import datetime
import time

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import os
import requests
import json
from PIL import Image
import base64
import io

# --- CONFIGURATION FOR GOOGLE FORMS LOGGING ---
# REPLACE THESE WITH YOUR ACTUAL VALUES FROM YOUR GOOGLE FORM
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfdtpLXu7Q8YZqQVfRRd-txzpqGdXIQQmhQ4nzbt943Hlf5-Q/formResponse"
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

PASSWORD_FOR_UPLOAD = "1234"

# --- Logging Function (Modified for Google Forms - now tracks up to 10 queries per session) ---
def log_activity_to_google_form(user_name, event_type, query_content=""):
    """
    Logs user activity to a Google Form, handling up to 10 queries per user session.
    """
    current_datetime = datetime.datetime.now()
    log_date = current_datetime.strftime("%Y-%m-%d")
    log_time = current_datetime.strftime("%H:%M:%S")

    # Initialize session state for logging if not present
    if "current_client_log_data" not in st.session_state:
        st.session_state.current_client_log_data = {
            "user_name": user_name,
            "log_date": log_date,
            "log_time": log_time,
            "queries": [""] * 10, # Initialize 10 empty slots for queries
            "query_count": 0 # Track how many queries have been made in this session
        }
        # For the "New Session" event, we want to capture the initial user details
        if event_type == "New Session":
            form_data = {
                FORM_ENTRY_IDS["user_name"]: user_name,
                FORM_ENTRY_IDS["log_date"]: log_date,
                FORM_ENTRY_IDS["log_time"]: log_time,
                FORM_ENTRY_IDS["event_type"]: event_type,
                FORM_ENTRY_IDS["query_1"]: "", # Initial queries are empty
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
                # st.toast("New session logged to Google Form.")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error logging new session: {e}")
            except Exception as e:
                st.error(f"Error logging new session: {e}")
            return # Don't process as a query event immediately

    # If it's a "Chat Query" event and we are within the first 10 queries
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
        # Only populate the specific query field that was just made
        query_key = f"query_{st.session_state.current_client_log_data['query_count']}" # query_1, query_2 etc.
        if query_key in FORM_ENTRY_IDS:
            form_data.update({FORM_ENTRY_IDS.get(query_key, ""): query_content})
        else:
            # Fallback for more than 10 queries, or if form structure changes
            form_data.update({FORM_ENTRY_IDS.get("event_type"): f"Query {st.session_state.current_client_log_data['query_count']} (Exceeded 10 tracked queries)"})
            form_data.update({FORM_ENTRY_IDS.get("query_1"): query_content}) # Put later queries into the first query field

        try:
            requests.post(GOOGLE_FORM_URL, data=form_data)
            # st.toast(f"Logged query {st.session_state.current_client_log_data['query_count']} to Google Form.")
        except requests.exceptions.RequestException as e:
            st.error(f"Network error logging query: {e}")
        except Exception as e:
            st.error(f"Error logging query: {e}")

    # For other event types (like "Image Analysis" or "Chat History Cleared")
    elif event_type != "New Session": # New Session handled above
        form_data = {
            FORM_ENTRY_IDS["user_name"]: user_name,
            FORM_ENTRY_IDS["log_date"]: log_date,
            FORM_ENTRY_IDS["log_time"]: log_time,
            FORM_ENTRY_IDS["event_type"]: event_type,
            FORM_ENTRY_IDS["query_1"]: query_content, # Put event content in Query 1 field
        }
        try:
            requests.post(GOOGLE_FORM_URL, data=form_data)
            # st.toast(f"Logged event '{event_type}' to Google Form.")
        except requests.exceptions.RequestException as e:
            st.error(f"Network error logging event: {e}")
        except Exception as e:
            st.error(f"Error logging event: {e}")

# Simple document processing
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF - super simple"""
    try:
        doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        all_text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add pages with text
                all_text += f"\n\n=== Page {page_num + 1} from {pdf_file.name} ===\n{text}"

        doc.close()
        return all_text
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {str(e)}")
        return ""

# Image processing for vision AI
def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_image_with_huggingface(image, question):
    """Analyze image using free Hugging Face API"""
    try:
        # Convert image to base64
        img_base64 = encode_image_to_base64(image)

        # Use Hugging Face's free inference API
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

        # Simple image captioning (free tier)
        response = requests.post(
            API_URL,
            headers={"Authorization": "Bearer hf_demo"},  # Using demo token
            json={"inputs": img_base64}
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get('generated_text', 'Unable to analyze image')
                return f"Image Analysis: {caption}\n\nBased on your question '{question}', this image appears to show: {caption}"
            else:
                return "I can see the image but couldn't generate a detailed description."
        else:
            return "Image analysis service temporarily unavailable. Please try again later."

    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def analyze_image_with_local_description(image, question):
    """Fallback: Basic image analysis without external APIs"""
    try:
        # Get basic image info
        width, height = image.size
        format_info = image.format or "Unknown"
        mode = image.mode

        # Simple analysis based on image properties
        analysis = f"""Image Analysis Results:

**Image Properties:**
- Dimensions: {width} x {height} pixels
- Format: {format_info}
- Color Mode: {mode}

**Your Question:** {question}

**Basic Analysis:** I can see this is a {width}x{height} pixel {format_info.lower()} image. To provide more detailed analysis about the content, you might want to:

1. **Describe what you see** in your question for better context
2. **Ask specific questions** about elements in the image
3. **Upload related documents** that might provide context

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

# Simple search function
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

# Split text into chunks
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
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

# Function to handle chat responses
def handle_chat_response(prompt):
    if "user_name" not in st.session_state or st.session_state.user_name is None:
        st.warning("Please tell me your name first!")
        return

    # Log the user's query to Google Form
    log_activity_to_google_form(st.session_state.user_name, "Chat Query", prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching through your documents..."):
            relevant_docs = search_documents(prompt)

            if relevant_docs:
                # Personalize the response with the user's name
                response_prefix = f"Hello {st.session_state.user_name}, based on your documents, here's what I found:\n\n"
                response = response_prefix

                for i, doc in enumerate(relevant_docs[:3], 1):  # Show top 3 results
                    lines = doc.split('\n')
                    source_line = next((line for line in lines if '===' in line and 'from' in line), '')

                    if source_line:
                        response += f"**Source {i}:** {source_line.replace('===', '').strip()}\n\n"

                    clean_doc = doc.replace(source_line, '').strip()
                    clean_doc = '\n'.join([line for line in clean_doc.split('\n') if not line.startswith('===')])

                    response += f"{clean_doc[:500]}{'...' if len(clean_doc) > 500 else ''}\n\n"
                    response += "---\n\n"

                st.write(response)
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                no_results_msg = f"Sorry {st.session_state.user_name}, I couldn't find relevant information in your uploaded documents. Please try:\n\n- Rephrasing your question\n- Using different keywords\n- Making sure your documents have been processed\n- Checking if the information exists in your uploaded files\n- Or try uploading an image if your question is about visual content!"
                st.write(no_results_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_results_msg})

# Main app
def main():
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Apply custom CSS for background color, header text color, and input border
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #870F26; /* Background color for the whole page */
            color: #FFFFFF; /* Default text color for the entire app */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF; /* All headings white */
        }
        p {
            color: #FFFFFF; /* Paragraph text white */
        }
        /* Make markdown elements (like st.write("---") or regular text) white */
        .stMarkdown {
            color: #FFFFFF;
        }

        /* Specifically target text within chat messages to be white, if not already */
        div[data-testid="stChatMessage"] p {
            color: #FFFFFF;
        }

        /* Ensure input labels are white */
        .stTextInput label, .stFileUploader label {
            color: #FFFFFF;
        }

        /* Keep the custom bold border for the text input */
        .stTextInput > div > div > input {
            border: 3px solid #870F26; /* Bold border for text inputs */
        }
        /* Specific border for the st.chat_input, might override generic stTextInput if needed */
        div[data-testid="stChatInput"] div[data-testid="stTextInput"] div[data-testid="stForm"] > div > div > input {
           border: 3px solid #870F26 !important; /* Specific bold border for chat input, important to ensure override */
        }
        /* Also apply border to the parent container of st.chat_input for visual consistency */
        div[data-testid="stChatInput"] {
            border: 3px solid #870F26; /* Border for the entire chat input container */
            border-radius: 0.5rem; /* Slightly rounded corners for the border */
            padding: 0.5rem; /* Some padding inside the border */
        }

        /* Ensure .stAlert (info, warning, error) boxes maintain their default styling */
        div[data-testid="stAlert"] {
            color: unset !important; /* Revert text color to browser default for alerts */
            background-color: unset !important; /* Revert background color */
        }
        div[data-testid="stAlert"] p {
             color: unset !important; /* Ensure paragraph text inside alerts is also default */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 AI Document Assistant")
    st.write("Upload PDF documents and images, then ask questions about their content!")

    # Initial greeting and name input
    if "user_name" not in st.session_state:
        st.session_state.user_name = None # Initialize to None if not present

    if st.session_state.user_name is None:
        user_name_input = st.text_input("Hey there, how shall I address you?")
        if user_name_input:
            st.session_state.user_name = user_name_input.strip()
            # Log the new session to Google Form
            log_activity_to_google_form(st.session_state.user_name, "New Session")
            st.session_state.messages = [] # Clear messages for new user
            st.success(f"Nice to meet you, {st.session_state.user_name}! You can now ask questions or upload documents.")
            time.sleep(1) # Give time for message to display
            st.rerun() # Rerun to remove the name input and proceed with the main app
        return # Stop execution until name is entered
    else:
        # If name is already set, display welcome message for existing user
        # Increased font size for the welcome message as well, and bolded the name
        st.markdown(f"<p style='font-size: 20px;'>Welcome back, <span style='font-size: 24px; font-weight: bold;'>{st.session_state.user_name}</span>! How can I help you today?</p>", unsafe_allow_html=True)

    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.header("📁 Document Management")

        # Password input for document upload
        password = st.text_input("Enter password to upload documents:", type="password", key="upload_password")

        if password == PASSWORD_FOR_UPLOAD:
            st.success("Password accepted! You can now upload documents.")
            # File uploader for PDFs
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload multiple PDF files to create your knowledge base"
            )

            # Process button
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
                if uploaded_files:
                    embedder, collection = setup_simple_search()

                    if embedder and collection:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        total_files = len(uploaded_files)
                        processed_chunks = 0

                        for file_idx, pdf_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {pdf_file.name}...")

                            # Extract text
                            text = extract_text_from_pdf(pdf_file)

                            if text.strip():
                                # Split into chunks
                                chunks = split_text_into_chunks(text)

                                # Add to search database
                                for chunk_idx, chunk in enumerate(chunks):
                                    try:
                                        collection.add(
                                            documents=[chunk],
                                            ids=[f"{pdf_file.name}_chunk_{chunk_idx}_{processed_chunks}"]
                                        )
                                        processed_chunks += 1
                                    except Exception as e:
                                        st.warning(f"Error processing chunk from {pdf_file.name}: {str(e)}")

                                # Update progress
                                progress_bar.progress((file_idx + 1) / total_files)

                        status_text.text("✅ Processing complete!")
                        st.success(f"Successfully processed {total_files} documents into {processed_chunks} searchable chunks!")

                        # Clear progress indicators after 2 seconds
                        time.sleep(2)
                        progress_bar.empty()
                        status_text.empty()
                else:
                    st.warning("Please upload at least one PDF file.")

            # Show some helpful info
            st.info("💡 **Tips:**\n- Upload multiple PDFs at once\n- Supported formats: PDF only\n- Processing may take a few minutes for large files")
        elif password: # Only show error if password was actually entered
            st.error("Incorrect password. Please try again.")

    with col2:
        st.header("💬 Ask Questions")

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message.get("image"):
                        st.image(message["image"], caption="Uploaded Image", use_column_width=True)
                    st.write(message["content"])

        # Add the three clickable buttons
        st.write("---") # Separator for better UI
        st.subheader("Quick Queries")

        if st.button("Give me a Summary of the Construction Handbook", use_container_width=True):
            handle_chat_response("Give me a summary of the construction handbook.")

        if st.button("What is the grounding resistance on a site", use_container_width=True):
            handle_chat_response("What is the grounding resistance on a site?")

        if st.button("What is the concrete strength on a tower foundation", use_container_width=True):
            handle_chat_response("What is the concrete strength on a tower foundation?")

        st.write("---") # Separator

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents or images..."):
            handle_chat_response(prompt)

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            # When chat is cleared, also reset user name to prompt again
            # Note: logging this specific event to Google Forms.
            log_activity_to_google_form(st.session_state.user_name, "Chat History Cleared")
            st.session_state.messages = []
            st.session_state.user_name = None
            # Reset the session log data to ensure a fresh start for the next user
            st.session_state.current_client_log_data = None
            st.rerun()

    with col3:
        st.header("🖼️ Image Analysis")

        # Image uploader
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload images for AI analysis"
        )

        if uploaded_image is not None:
            # Display the image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Image analysis input
            image_question = st.text_input(
                "Ask about this image:",
                placeholder="What do you see in this image?",
                help="Ask specific questions about the image content"
            )

            if st.button("🔍 Analyze Image", use_container_width=True):
                if image_question:
                    with st.spinner("Analyzing image..."):
                        # Try advanced analysis first, fallback to basic
                        try:
                            analysis = analyze_image_with_huggingface(image, image_question)
                        except Exception as e: # Catch specific exceptions or general one here
                            st.warning(f"Hugging Face analysis failed: {e}. Falling back to basic analysis.")
                            analysis = analyze_image_with_local_description(image, image_question)

                        # Log the image analysis event
                        log_activity_to_google_form(st.session_state.user_name, "Image Analysis", image_question)

                        # Add to chat
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

        # Image analysis tips
        st.info("🖼️ **Image Tips:**\n- Supports: PNG, JPG, JPEG, GIF, BMP\n- Ask specific questions\n- Works with diagrams, charts, photos\n- Combines with document search")

    # Copyright remark at the bottom
    st.markdown("---")
    st.markdown("Copyright by Werner Ton 2025", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
