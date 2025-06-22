import streamlit as st
import requests
import time
import asyncio
from datetime import datetime
import os
import json

# Page configuration with 3D effects
st.set_page_config(
    page_title="🚀 Gemini Research Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern 3D CSS with glassmorphism and animations
st.markdown("""
    <style>
    :root {
        --primary: #6366F1;
        --primary-light: #818CF8;
        --secondary: #10B981;
        --accent: #EC4899;
        --dark: #0F172A;
        --darker: #020617;
        --light: #F8FAFC;
        --glass: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--darker), var(--dark));
        color: var(--light);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
        overflow-x: hidden;
    }
    
    /* 3D Background Elements */
    .bg-element {
        position: fixed;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.15;
        z-index: -1;
    }
    
    .bg-1 {
        width: 500px;
        height: 500px;
        background: var(--primary);
        top: -200px;
        right: -200px;
    }
    
    .bg-2 {
        width: 700px;
        height: 700px;
        background: var(--secondary);
        bottom: -300px;
        left: -300px;
    }
    
    .bg-3 {
        width: 300px;
        height: 300px;
        background: var(--accent);
        top: 40%;
        left: 10%;
    }
    
    /* Header with 3D Effect */
    .header {
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transform: perspective(1000px) rotateX(5deg);
        border: 1px solid var(--glass-border);
    }
    
    .header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, var(--accent), transparent);
    }
    
    .header h1 {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(45deg, #ffffff, #e0f7fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        letter-spacing: -0.5px;
        transform: translateZ(20px);
    }
    
    .header p {
        font-size: 1.2rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    /* 3D Card Effects */
    .card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.8rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transform-style: preserve-3d;
        transform: translateZ(0);
    }
    
    .card:hover {
        transform: translateY(-8px) translateZ(10px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .card-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--primary-light);
    }
    
    /* 3D Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 15px rgba(99, 102, 241, 0.4) !important;
        position: relative;
        overflow: hidden;
        transform: translateZ(0);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.03) !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.6) !important;
    }
    
    .stButton>button:active {
        transform: translateY(2px) !important;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Clear button styling */
    .clear-btn {
        background: linear-gradient(90deg, #EF4444, #F87171) !important;
        box-shadow: 0 6px 15px rgba(239, 68, 68, 0.4) !important;
    }
    
    .clear-btn:hover {
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.6) !important;
    }
    
    /* AI Response with 3D Effect */
    .ai-response {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border-left: 4px solid var(--primary-light);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.8rem 0;
        animation: fadeIn 0.6s ease;
        transform: perspective(800px) rotateY(-1deg);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px) perspective(800px) rotateY(-5deg); }
        to { opacity: 1; transform: translateY(0) perspective(800px) rotateY(-1deg); }
    }
    
    .ai-response-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        color: var(--primary-light);
    }
    
    /* Source Tags with 3D Effect */
    .source-tag {
        background: rgba(99, 102, 241, 0.2);
        color: var(--primary-light);
        padding: 8px 16px;
        border-radius: 20px;
        margin: 6px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        transform: translateZ(0);
    }
    
    .source-tag:hover {
        transform: translateY(-3px) translateZ(5px);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.3);
        background: rgba(99, 102, 241, 0.3);
    }
    
    /* Metrics with 3D Effect */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin-top: 1.5rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        transform: translateZ(0);
        border: 1px solid var(--glass-border);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card:hover {
        transform: translateY(-5px) translateZ(5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(45deg, var(--primary-light), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Status Indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px 0;
        backdrop-filter: blur(4px);
        border: 1px solid var(--glass-border);
    }
    
    .status-online {
        background: linear-gradient(90deg, var(--secondary), #10B981);
        color: white;
    }
    
    .status-offline {
        background: linear-gradient(90deg, #EF4444, #F87171);
        color: white;
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        color: white;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 1000;
        transform: translateZ(0);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .fab:hover {
        transform: scale(1.1) rotate(15deg) translateZ(5px);
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.7);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--primary), var(--primary-light));
        border-radius: 4px;
    }
    
    /* Chat History Items */
    .chat-item {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid var(--glass-border);
    }
    
    .chat-item:hover {
        background: rgba(30, 41, 59, 0.8);
        transform: translateX(5px);
    }
    
    .chat-question {
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .chat-answer {
        font-size: 0.9rem;
        opacity: 0.8;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    .chat-time {
        font-size: 0.75rem;
        opacity: 0.6;
        text-align: right;
        margin-top: 0.5rem;
    }
    
    /* Processing Animation */
    @keyframes processing {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .processing-bar {
        height: 4px;
        background: linear-gradient(90deg, transparent, var(--primary-light), transparent);
        width: 100%;
        position: relative;
        overflow: hidden;
        border-radius: 2px;
        margin: 1rem 0;
    }
    
    .processing-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        animation: processing 1.5s infinite;
    }
    
    /* Error Message Styling */
    .error-message {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), transparent);
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    
    <div class="bg-element bg-1"></div>
    <div class="bg-element bg-2"></div>
    <div class="bg-element bg-3"></div>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize session state
def init_session_state():
    defaults = {
        'processed': False,
        'file_name': None,
        'processing': False,
        'last_question': '',
        'last_answer': '',
        'last_sources': [],
        'backend_status': None,
        'error_message': None,
        'processing_time': 0,
        'chat_history': [],
        'confidence_score': None,
        'current_file': None,
        'processing_progress': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Check backend connection
async def check_backend_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            st.session_state.backend_status = response.json()
            return True
        st.session_state.backend_status = None
        return False
    except:
        st.session_state.backend_status = None
        return False

# Process document with visual feedback
def process_document(uploaded_file):
    try:
        st.session_state.processing = True
        st.session_state.processing_progress = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress for better UX
        steps = [
            "📄 Uploading document...",
            "🔍 Extracting text from PDF...",
            "✂️ Splitting into chunks...",
            "🧠 Generating embeddings...",
            "💾 Storing in vector database...",
            "✅ Finalizing processing..."
        ]
        
        for i, step in enumerate(steps):
            status_text.info(step)
            st.session_state.processing_progress = (i + 1) * (100 // len(steps))
            progress_bar.progress(st.session_state.processing_progress)
            time.sleep(0.7)
        
        # Actual API call
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(
            f"{API_BASE_URL}/upload", 
            files=files,
            timeout=120
        )
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.processed = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.processing_time = result.get('processing_time', 0)
            st.session_state.current_file = uploaded_file
            st.session_state.error_message = None
            st.success("✅ Document processed successfully!")
            return True
        else:
            error_detail = response.json().get('detail', 'Processing failed')
            st.session_state.error_message = f"❌ Error: {error_detail}"
            return False
    except Exception as e:
        st.session_state.error_message = f"❌ Unexpected error: {str(e)}"
        return False
    finally:
        st.session_state.processing = False
        st.session_state.processing_progress = 0

# Ask question with visual feedback
def ask_question(question):
    try:
        st.session_state.asking = True
        with st.spinner(""):
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <div class="processing-bar"></div>
                <div style="margin-top: 1rem; font-size: 1.2rem; color: var(--primary-light);">
                    🤖 Gemini is analyzing your question...
                </div>
                <div style="margin-top: 1rem; opacity: 0.7;">
                    • Searching document chunks<br>
                    • Analyzing context relevance<br>
                    • Generating comprehensive response
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": question, "max_chunks": 5},
                timeout=60
            )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.last_question = question
            st.session_state.last_answer = data['answer']
            st.session_state.last_sources = data.get('sources', [])
            st.session_state.confidence_score = data.get('confidence_score')
            st.session_state.processing_time = data.get('processing_time', 0)
            st.session_state.error_message = None
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': question,
                'answer': data['answer'],
                'timestamp': datetime.now(),
                'sources': data.get('sources', [])
            })
            return data
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.session_state.error_message = f"❌ Error: {error_detail}"
            return None
    except Exception as e:
        st.session_state.error_message = f"❌ Unexpected error: {str(e)}"
        return None
    finally:
        st.session_state.asking = False

# Clear all data properly
def clear_all_data():
    try:
        # API call to clear backend data
        response = requests.delete(f"{API_BASE_URL}/clear")
        
        # Reset session state
        keys_to_reset = [
            'processed', 'file_name', 'last_question', 
            'last_answer', 'last_sources', 'processing_time',
            'chat_history', 'confidence_score', 'current_file',
            'error_message'
        ]
        
        for key in keys_to_reset:
            if key == 'chat_history':
                st.session_state[key] = []
            elif key == 'processed':
                st.session_state[key] = False
            else:
                st.session_state[key] = None
        
        st.session_state.processing = False
        st.success("♻️ All data cleared successfully!")
        
    except Exception as e:
        st.session_state.error_message = f"❌ Error clearing data: {str(e)}"

# Initialize session
init_session_state()

# Header with 3D effect
st.markdown("""
    <div class="header">
        <h1>🚀 GEMINI RESEARCH LAB</h1>
        <p>3D Document Intelligence Powered by Google Gemini AI</p>
    </div>
""", unsafe_allow_html=True)

# Floating action button
st.markdown("""
    <div class="fab" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
        ↑
    </div>
""", unsafe_allow_html=True)

# Check backend connection
backend_connected = asyncio.run(check_backend_connection())

if not backend_connected:
    st.error("🔌 Backend Disconnected - Start FastAPI server first")
    st.code("python app.py")
    st.stop()

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    # Document upload section with 3D card
    with st.container():
        st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span>📤</span>
                    <span>Document Upload</span>
                </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload research papers or documents for analysis",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"**Selected:** {uploaded_file.name} ({file_size:.2f} MB)")
            
            col_btn1, col_btn2 = st.columns([2, 1])
            with col_btn1:
                if st.button("🚀 PROCESS DOCUMENT", use_container_width=True, 
                             disabled=st.session_state.processing):
                    process_document(uploaded_file)
            with col_btn2:
                if st.session_state.processed:
                    if st.button("🗑️ CLEAR DOCUMENT", use_container_width=True, 
                                key="clear_doc", on_click=clear_all_data,
                                type="secondary"):
                        pass
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card
    
    # Question interface
    if st.session_state.processed:
        with st.container():
            st.markdown("""
                <div class="card">
                    <div class="card-header">
                        <span>🧠</span>
                        <span>Ask Your Research Question</span>
                    </div>
            """, unsafe_allow_html=True)
            
            # Suggested questions with 3D effect
            suggestions = [
                "What are the main findings?",
                "Summarize the methodology",
                "What are the conclusions?",
                "List key limitations",
                "Future research directions"
            ]
            
            st.write("💡 Try asking:")
            
            # Create columns for suggestions
            cols = st.columns(len(suggestions))
            for i, col in enumerate(cols):
                with col:
                    if st.button(suggestions[i], use_container_width=True, 
                                key=f"suggest_{i}"):
                        st.session_state.question_input = suggestions[i]
                        ask_question(suggestions[i])
            
            # Question input
            question = st.text_area(
                "Your question:",
                placeholder="What insights can you extract from this document?",
                height=120,
                key="question_input"
            )
            
            if st.button("🔍 ASK GEMINI", disabled=not question.strip() or st.session_state.asking, 
                         use_container_width=True):
                ask_question(question)
            
            # Display response
            if st.session_state.last_answer:
                st.markdown("""
                    <div class="ai-response">
                        <div class="ai-response-header">
                            <span>🤖</span>
                            <h3>Gemini Response</h3>
                        </div>
                        <div style="margin-bottom: 1rem; font-weight: 600;">
                            {question}
                        </div>
                        <div>
                            {answer}
                        </div>
                    </div>
                """.format(
                    question=st.session_state.last_question,
                    answer=st.session_state.last_answer
                ), unsafe_allow_html=True)
                
                if st.session_state.last_sources:
                    st.markdown("**📚 Source References**")
                    for src in st.session_state.last_sources:
                        st.markdown(f"<div class='source-tag'>📄 {src}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close card

    # Error message display
    if st.session_state.error_message:
        st.markdown(f"""
            <div class="error-message">
                {st.session_state.error_message}
            </div>
        """, unsafe_allow_html=True)

with col2:
    # System status with 3D card
    with st.container():
        st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span>📊</span>
                    <span>System Status</span>
                </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.backend_status:
            status = st.session_state.backend_status
            
            st.markdown("""
                <div class="status-indicator status-online">
                    <span>🟢</span>
                    <span>ONLINE</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Document status
            doc_status = "📄 READY" if status['status'] == 'ready' else "📄 WAITING"
            st.markdown(f"**Document Status:** {doc_status}")
            
            if st.session_state.file_name:
                st.markdown(f"**Current File:** {st.session_state.file_name}")
            
            # AI Model
            st.markdown(f"**AI Model:** {status['ai_model']}")
            
            # Metrics with 3D effect
            if st.session_state.processed:
                st.divider()
                st.markdown("**⚙️ Performance Metrics**")
                
                st.markdown("""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{processing_time:.1f}s</div>
                            <div>Processing Time</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{confidence}%</div>
                            <div>AI Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{chunks}</div>
                            <div>Content Chunks</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{size:.1f}K</div>
                            <div>Text Size</div>
                        </div>
                    </div>
                """.format(
                    processing_time=st.session_state.processing_time,
                    confidence=int(st.session_state.confidence_score * 100) if st.session_state.confidence_score else 95,
                    chunks=st.session_state.backend_status['performance_metrics'].get('chunks_processed', 0),
                    size=st.session_state.backend_status['performance_metrics'].get('document_size', 0)/1000
                ), unsafe_allow_html=True)
        
        if st.button("🔄 REFRESH STATUS", use_container_width=True):
            asyncio.run(check_backend_connection())
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card
    
    # Chat history with 3D card
    if st.session_state.chat_history:
        with st.container():
            st.markdown("""
                <div class="card">
                    <div class="card-header">
                        <span>💬</span>
                        <span>Recent Questions</span>
                    </div>
            """, unsafe_allow_html=True)
            
            for chat in reversed(st.session_state.chat_history[-4:]):
                st.markdown(f"""
                    <div class="chat-item" onclick="document.getElementById('question_input').value = '{chat['question']}';">
                        <div class="chat-question">{chat['question']}</div>
                        <div class="chat-answer">{chat['answer'][:100]}...</div>
                        <div class="chat-time">{chat['timestamp'].strftime('%Y-%m-%d %H:%M')}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close card
    
    # Technology stack with 3D card
    with st.container():
        st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span>🛠️</span>
                    <span>Technology Stack</span>
                </div>
                <div style="line-height: 2;">
                    <div>• <strong>AI Model:</strong> Gemini 1.5 Flash</div>
                    <div>• <strong>Embeddings:</strong> MiniLM-L6-v2</div>
                    <div>• <strong>Vector DB:</strong> ChromaDB</div>
                    <div>• <strong>Backend:</strong> FastAPI + Async</div>
                    <div>• <strong>Frontend:</strong> Streamlit</div>
                    <div>• <strong>Processing:</strong> RAG Architecture</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("🚀 Gemini Research Lab • Powered by Google Gemini AI • Built with FastAPI & Streamlit • 3D UI Experience")

# Custom JavaScript for smooth interactions
st.markdown("""
    <script>
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Handle clear button confirmation
    const clearButtons = document.querySelectorAll('button[kind="secondary"]');
    clearButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (!confirm('Are you sure you want to clear all data?')) {
                event.stopPropagation();
            }
        });
    });
    
    // Parallax effect for background elements
    window.addEventListener('scroll', function() {
        const scrollY = window.scrollY;
        document.querySelector('.bg-1').style.transform = `translateY(${scrollY * 0.2}px)`;
        document.querySelector('.bg-2').style.transform = `translateY(${scrollY * 0.1}px)`;
        document.querySelector('.bg-3').style.transform = `translateY(${scrollY * 0.15}px)`;
    });
    </script>
""", unsafe_allow_html=True)