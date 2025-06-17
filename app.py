# app.py - Clean RAG Chatbot with Design Principles Applied

import streamlit as st
import time
import os
import tempfile
import shutil
from pathlib import Path
from vectors import EmbeddingsManager  
from chatbot import ChatbotManager    
from dotenv import load_dotenv, find_dotenv
import sqlite3
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DESIGN PRINCIPLES APPLIED:

# 1. SECURITY: File size limits and validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx', 'pptx']

# 2. COST: Resource limits and session controls - Adjusted for document analysis
MAX_INTERACTIONS_PER_SESSION = 50
MAX_TOKENS_PER_SESSION = 100000  # Increased for document analysis

# 3. GUARDRAILS: Automated policy enforcement
def validate_file_upload(uploaded_file):
    """SECURITY: Validate file uploads with multiple checks"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB"
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # SECURITY: Check for malicious file names
    if any(char in uploaded_file.name for char in ['<', '>', '..', '/', '\\']):
        return False, "Invalid characters in filename"
    
    return True, "Valid file"

# 4. COST: Efficient logging with size limits
def log_interaction(session_id, user_input, bot_response, is_flagged=False, flag_reason=None):
    """COST OPTIMIZATION: Efficient logging with size limits"""
    try:
        conn = sqlite3.connect("chat_logs.db", timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                user_input TEXT,
                bot_response TEXT,
                is_flagged BOOLEAN,
                flag_reason TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO chat_logs (id, session_id, timestamp, user_input, bot_response, is_flagged, flag_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            session_id,
            datetime.now().isoformat(),
            user_input[:1000],  # COST: Limit input length
            bot_response[:8000],  # COST: Limit response length
            is_flagged,
            flag_reason
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
    finally:
        try:
            conn.close()
        except:
            pass

# Load environment variables
load_dotenv(find_dotenv())

# 5. AGILITY: Simple page configuration
st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# AGILITY: Simple CSS without complex HTML
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stAlert > div { padding: 0.75rem 1rem; }
    .chat-message { 
        padding: 1rem; 
        margin: 0.5rem 0; 
        border-radius: 0.5rem; 
        border-left: 4px solid #4CAF50;
    }
    .user-message { 
        background-color: #e3f2fd; 
        border-left-color: #2196f3; 
        margin-left: 20%;
    }
    .assistant-message { 
        background-color: #f3e5f5; 
        border-left-color: #9c27b0; 
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# AGILITY: Simple session state initialization
def initialize_session_state():
    """AGILITY: Quick session initialization with defaults"""
    defaults = {
        'temp_pdf_path': None,
        'session_id': str(uuid.uuid4()),
        'pdf_saved': False,
        'embeddings_saved': False,
        'chatbot_manager': None,
        'messages': [],
        'document_info': {},
        'processing': False,
        'interaction_count': 0,
        'total_tokens_used': 0,
        'error_count': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Header
st.title("📚 Document RAG Chatbot")
st.markdown("Upload documents and chat with AI")

# GUARDRAILS: Sidebar with controls and monitoring
with st.sidebar:
    st.header("🔧 Configuration & Controls")
    
    # GUARDRAILS: Environment validation
    st.subheader("🌍 System Status")
    env_checks = {
        "QDRANT_URL": bool(os.getenv('QDRANT_URL')),
        "GROQ_API_KEY": bool(os.getenv('GROQ_API_KEY')),
        "QDRANT_API_KEY": bool(os.getenv('QDRANT_API_KEY'))
    }
    
    for key, status in env_checks.items():
        icon = "✅" if status else "❌"
        st.write(f"{icon} {key}")
    
    if not all(env_checks.values()):
        st.error("⚠️ Missing required environment variables!")
    
    st.divider()
    
    # AGILITY: Dynamic configuration
    st.subheader("⚙️ Model Settings")
    llm_model = st.selectbox(
        "LLM Model",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 1000, 8000, 4000, 500)
    
    st.subheader("🔍 Retrieval Settings")
    retrieval_k = st.slider("Documents to Retrieve", 1, 10, 5, 1)
    score_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.05)
    
    st.subheader("📝 Processing Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
    
    # GUARDRAILS: Security controls
    st.subheader("🛡️ Security & Guardrails")
    enable_content_filter = st.toggle("Content Filtering", value=True)
    enable_pii_detection = st.toggle("PII Detection", value=True)
    max_interactions = st.slider("Max Interactions", 10, 100, MAX_INTERACTIONS_PER_SESSION, 5)
    
    st.divider()
    
    # COST: Session monitoring
    st.subheader("📊 Session Monitor")
    st.metric("Interactions", f"{st.session_state.interaction_count}/{max_interactions}")
    st.metric("Errors", st.session_state.error_count)
    st.metric("Tokens Used", st.session_state.total_tokens_used)
    
    # Show session info for debugging
    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    if st.session_state.get('embeddings_saved'):
        collection_name = f"doc_collection_{st.session_state.session_id[:8]}"
        st.caption(f"Vector Collection: {collection_name}")
    
    # AGILITY: Quick session reset with proper cleanup
    if st.button("🔄 Reset Session", type="secondary"):
        # Clean up the current session's vector collection
        if st.session_state.get('chatbot_manager'):
            try:
                collection_name = f"doc_collection_{st.session_state.session_id[:8]}"
                st.session_state['chatbot_manager'].qdrant_client.delete_collection(collection_name)
                logger.info(f"Cleaned up collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Collection cleanup failed: {e}")
        
        # Reset all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # SECURITY: File upload with validation
    st.subheader("📤 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=ALLOWED_EXTENSIONS,
        help=f"Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
    )
    
    if uploaded_file is not None:
        is_valid, message = validate_file_upload(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {message}")
        else:
            try:
                # SECURITY: Safe file handling
                temp_dir = tempfile.mkdtemp()
                safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in ".-_")
                temp_file_path = os.path.join(temp_dir, safe_filename)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state['temp_pdf_path'] = temp_file_path
                st.session_state['pdf_saved'] = True
                st.session_state['document_info'] = {
                    'name': safe_filename,
                    'size': f"{uploaded_file.size / 1024:.2f} KB",
                    'type': uploaded_file.type
                }
                
                st.success(f"✅ Uploaded: {safe_filename}")
                
            except Exception as e:
                logger.error(f"Upload error: {e}")
                st.error(f"❌ Upload failed: {str(e)}")
    
    # SCALABILITY: Document processing
    if st.session_state['pdf_saved'] and not st.session_state['processing']:
        st.subheader("⚙️ Process Document")
        
        # Show current session collection name for debugging
        current_collection = f"doc_collection_{st.session_state.session_id[:8]}"
        st.caption(f"Collection: {current_collection}")
        
        if st.button("🚀 Process Document", disabled=st.session_state['embeddings_saved'], type="primary"):
            st.session_state['processing'] = True
            
            with st.spinner("Processing document..."):
                try:
                    # SCALABILITY: Efficient processing with progress tracking
                    progress_bar = st.progress(0)
                    
                    progress_bar.progress(25)
                    embeddings_manager = EmbeddingsManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url=os.getenv('QDRANT_URL'),
                        collection_name=f"doc_collection_{st.session_state.session_id[:8]}",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    progress_bar.progress(50)
                    result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                    
                    progress_bar.progress(75)
                    st.session_state['chatbot_manager'] = ChatbotManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        llm_model=llm_model,
                        llm_temperature=temperature,
                        max_tokens=max_tokens,
                        qdrant_url=os.getenv('QDRANT_URL'),
                        collection_name=f"doc_collection_{st.session_state.session_id[:8]}",
                        retrieval_k=retrieval_k,
                        score_threshold=score_threshold
                    )
                    
                    progress_bar.progress(100)
                    st.session_state['embeddings_saved'] = True
                    st.session_state['processing'] = False
                    
                    st.success("🎉 Document processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.session_state['processing'] = False
                    st.session_state['error_count'] += 1
                    logger.error(f"Processing error: {e}")
                    st.error(f"❌ Processing failed: {str(e)}")

# Main Chat Interface
st.header("💬 Chat with Your Document")

# GUARDRAILS: Check session limits
if st.session_state.interaction_count >= max_interactions:
    st.warning(f"⚠️ Maximum interactions ({max_interactions}) reached. Please reset session.")
elif not st.session_state['embeddings_saved']:
    # Welcome screen
    st.info("👆 Please upload and process a document first using the sidebar controls.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Supported Formats", "PDF, DOCX, TXT, PPTX")
    with col2:
        st.metric("🔒 Security", "PII Detection, Content Filter")
    with col3:
        st.metric("⚡ Features", "Real-time Processing")
    with col4:
        st.metric("📊 Monitoring", "Usage Tracking")
    
else:
    # AGILITY: Display chat messages
    if st.session_state['messages']:
        for msg in st.session_state['messages']:
            if msg['role'] == 'user':
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # GUARDRAILS: Show content filtering alerts
                    if msg.get('is_flagged'):
                        st.warning(f"⚠️ Content filtered: {msg.get('flag_reason', 'Unknown')}")
                    
                    # AGILITY: Source information
                    if msg.get('sources'):
                        with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                            for i, source in enumerate(msg['sources'], 1):
                                st.markdown(f"""
                                **Source {i}:** {source.get('file_name', 'Unknown')}  
                                **Page:** {source.get('page', 'N/A')} | **Words:** {source.get('word_count', 'N/A')}  
                                **Preview:** {source.get('content_preview', 'No preview')[:200]}...
                                """)
    else:
        st.info("🎯 Start a conversation by asking a question about your document below!")

# AGILITY: Chat input at bottom
if st.session_state['embeddings_saved'] and st.session_state.interaction_count < max_interactions:
    user_input = st.chat_input("Ask about your document...")
    
    if user_input:
        st.session_state.interaction_count += 1
        
        with st.spinner("🤔 Thinking..."):
            try:
                # SCALABILITY: Process response with guardrails
                response_data = st.session_state['chatbot_manager'].get_response(
                    user_input,
                    enable_content_filter=enable_content_filter,
                    enable_pii_detection=enable_pii_detection
                )
                
                # COST: Track token usage
                tokens_used = response_data.get('tokens_used', 0)
                st.session_state.total_tokens_used += tokens_used
                
                # Add messages to history
                st.session_state['messages'].append({
                    "role": "user", 
                    "content": user_input
                })
                st.session_state['messages'].append({
                    "role": "assistant",
                    "content": response_data.get('answer', 'No response generated'),
                    "sources": response_data.get('sources', []),
                    "is_flagged": response_data.get('is_flagged', False),
                    "flag_reason": response_data.get('flag_reason')
                })
                
                # COST: Efficient logging
                log_interaction(
                    session_id=st.session_state['session_id'],
                    user_input=user_input,
                    bot_response=response_data.get('answer', ''),
                    is_flagged=response_data.get('is_flagged', False),
                    flag_reason=response_data.get('flag_reason')
                )
                
            except Exception as e:
                st.session_state['error_count'] += 1
                logger.error(f"Chat error: {e}")
                st.session_state['messages'].append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": [],
                    "is_flagged": True,
                    "flag_reason": "System error"
                })
        
        st.rerun()

# COST: Cleanup function for resource management
def cleanup_temp_files():
    """COST OPTIMIZATION: Clean up temporary files and vector collections"""
    # Clean up temporary files
    if st.session_state.get('temp_pdf_path') and os.path.exists(st.session_state['temp_pdf_path']):
        try:
            temp_dir = os.path.dirname(st.session_state['temp_pdf_path'])
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    # Clean up session-specific vector collection
    if st.session_state.get('chatbot_manager'):
        try:
            collection_name = f"doc_collection_{st.session_state.session_id[:8]}"
            st.session_state['chatbot_manager'].qdrant_client.delete_collection(collection_name)
            logger.info(f"Cleaned up vector collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Vector collection cleanup failed: {e}")

# Register cleanup
import atexit
atexit.register(cleanup_temp_files)