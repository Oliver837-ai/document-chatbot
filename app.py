import streamlit as st
import pymupdf  # for PDFs
from sentence_transformers import SentenceTransformer

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

# Simple document processing
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF - super simple"""
    try:
        doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        all_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
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
            headers={"Authorization": f"Bearer hf_demo"},  # Using demo token
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

# Main app
def main():
    st.set_page_config(
        page_title="AI Document & Vision Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Document & Vision Assistant")
    st.write("Upload PDF documents and images, then ask questions about their content!")
    
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.header("📁 Document Management")
        
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
                    import time
                    time.sleep(2)
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.warning("Please upload at least one PDF file.")
        
        # Show some helpful info
        st.info("💡 **Tips:**\n- Upload multiple PDFs at once\n- Supported formats: PDF only\n- Processing may take a few minutes for large files")

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
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents or images..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching through your documents..."):
                    relevant_docs = search_documents(prompt)
                    
                    if relevant_docs:
                        response = f"Based on your documents, here's what I found:\n\n"
                        
                        for i, doc in enumerate(relevant_docs[:3], 1):  # Show top 3 results
                            # Extract source info from the document
                            lines = doc.split('\n')
                            source_line = next((line for line in lines if '===' in line and 'from' in line), '')
                            
                            if source_line:
                                response += f"**Source {i}:** {source_line.replace('===', '').strip()}\n\n"
                            
                            # Clean up the document text
                            clean_doc = doc.replace(source_line, '').strip()
                            clean_doc = '\n'.join([line for line in clean_doc.split('\n') if not line.startswith('===')])
                            
                            response += f"{clean_doc[:500]}{'...' if len(clean_doc) > 500 else ''}\n\n"
                            response += "---\n\n"
                        
                        st.write(response)
                        
                        # Add to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        no_results_msg = "I couldn't find relevant information in your uploaded documents. Please try:\n\n- Rephrasing your question\n- Using different keywords\n- Making sure your documents have been processed\n- Checking if the information exists in your uploaded files\n- Or try uploading an image if your question is about visual content!"
                        st.write(no_results_msg)
                        st.session_state.messages.append({"role": "assistant", "content": no_results_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
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
                        except:
                            analysis = analyze_image_with_local_description(image, image_question)
                        
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

if __name__ == "__main__":
    main()
