@echo off
REM Setup script for Project Samarth (Windows)
REM Run this to set up the complete RAG system

echo 🌾 Project Samarth - Setup Script
echo ==================================
echo.

REM Step 1: Install dependencies
echo 📦 Step 1: Installing dependencies...
pip install -r requirements.txt
echo ✅ Dependencies installed
echo.

REM Step 2: Preprocess data
echo 📊 Step 2: Preprocessing and merging datasets...
python data_preprocessing.py
echo.

REM Step 3: Build RAG corpus
echo 📚 Step 3: Building RAG corpus...
python rag_corpus_builder.py
echo.

REM Step 4: Create vector store
echo 🔍 Step 4: Creating FAISS vector store...
python rag_vector_store.py
echo.

echo ==================================
echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Set your Gemini API key: set GEMINI_API_KEY=your-api-key
echo 2. Run the Streamlit app: streamlit run app.py
echo 3. Enter your API key in the sidebar and click 'Initialize RAG System'
echo.

pause

