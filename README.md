# 🚀 AI Research Assistant - Intelligent Document Analysis


The **AI Research Assistant** is a cutting-edge platform that transforms how you interact with academic documents and research papers. Powered by **Google's Gemini AI**, this application allows you to upload PDF documents and ask complex questions to extract valuable insights using advanced **Retrieval-Augmented Generation (RAG)** technology. Featuring a stunning **3D UI** with immersive visual effects, it revolutionizes document analysis into an engaging and efficient experience.

---

## ✨ Features

- **3D Immersive UI**: Glassmorphism design with floating elements and parallax effects  
- **Document Intelligence**: Upload and analyze PDF documents with AI-powered insights  
- **Gemini AI Integration**: Powered by Google's latest Gemini models  
- **Smart Q&A**: Ask complex questions about your documents and get precise answers  
- **Source Referencing**: See exactly which document sections support each answer  
- **Performance Metrics**: Track processing times and AI confidence scores  
- **Chat History**: Review previous questions and answers  
- **Responsive Design**: Works beautifully on desktop and mobile devices  

---

## 🛠️ Technology Stack

### 🔙 Backend:
- Python 3.12  
- FastAPI  
- Google Gemini AI (`google-generativeai`)  
- LangChain  
- ChromaDB (Vector Database)  
- Hugging Face Embeddings  

### 🎨 Frontend:
- Streamlit  
- Custom 3D CSS Animations  
- Glassmorphism UI Design  

### ☁️ Infrastructure:
- Uvicorn ASGI server  
- Async processing  
- Efficient document chunking and embedding  

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.12  
- Google Gemini API key ([Get your free API key](https://makersuite.google.com/app/apikey))  
- Git  

### 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Research-Assistant.git
cd AI-Research-Assistant
```
Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```
Install dependencies:
```bash
pip install -r requirements.txt
```

🔐 Set up environment variables
Create a .env file in the project root and add your Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```
▶️ Running the Application
Start the backend server:
```bash
python app.py

```

Start the frontend (in a new terminal):
```bash
streamlit run streamlit_app.py
```

### 🌐 Access the Application
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:8501

### 🖥️ Usage Guide
📄 Upload a Document:
- Click "Choose a PDF file" to upload a research paper or document
- Click "PROCESS DOCUMENT" to analyze the document

### ❓ Ask Questions:
- Type your question in the input field
- Use suggested questions for quick insights
- Click "ASK GEMINI" to get answers

### 📊 Review Results:
- View AI-generated answers with supporting sources
- Check performance metrics and confidence scores
- Explore chat history of previous questions

### 🧹 Clear Data:
- Click "CLEAR DOCUMENT" to remove current document and reset the session

### 🤝 Contributing
We welcome contributions!
To contribute:

### Fork the project

- Create your feature branch:
```bash
git checkout -b feature/AmazingFeature
```
- Commit your changes:

```bash
git commit -m 'Add some AmazingFeature'
```
- Push to the branch:
```bash
git push origin feature/AmazingFeature
```
- Open a Pull Request

### 📜 License
Distributed under the MIT License.
See LICENSE for more information.


