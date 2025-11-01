# 🌾 Project Samarth - AI Q&A System for Agricultural & Climate Data

A complete **end-to-end RAG (Retrieval-Augmented Generation)** system for querying Indian agricultural and climate datasets using Google Gemini LLM. Built with Streamlit, FAISS vector search, and sentence transformers.

---

## 🎯 Project Overview

Project Samarth enables users to ask natural language questions about:
- **Indian agriculture** (crop production, yield, districts, states)
- **Climate data** (rainfall patterns, IMD subdivisions)
- **Agricultural-climate relationships** (rainfall vs yield, production trends)

The system uses **RAG architecture** to ground answers on real data from the Ministry of Agriculture and IMD (Indian Meteorological Department).

---

## ✨ Features

- 🧠 **RAG-powered Q&A**: Answers grounded on real agricultural and climate data
- 🔍 **Semantic Search**: FAISS vector database for fast similarity search
- 🤖 **Google Gemini Integration**: Uses Gemini 2.0 Flash or Gemini 1.5 Pro
- 📊 **Interactive Visualizations**: Auto-generated charts for trends and comparisons
- 💬 **Streamlit Web UI**: Modern, responsive interface
- 📚 **Source Attribution**: Shows retrieved documents with metadata
- 🔄 **Answer Caching**: Faster responses for repeated queries
- 📈 **Data Explorer**: Browse and filter merged datasets

---

## 📁 Project Structure

```
project_samarth/
│
├── app.py                    # Streamlit web application
├── rag_corpus_builder.py     # Converts CSV to RAG corpus (JSONL)
├── rag_vector_store.py       # Creates FAISS vector database
├── rag_gemini_qa.py          # RAG Q&A system with Gemini
├── data_preprocessing.py      # Merges crop and rainfall datasets
│
├── data/
│   ├── India Agriculture Crop Production.csv  # Crop production dataset
│   ├── Sub_Division_IMD_2017.csv            # Rainfall dataset (IMD)
│   └── merged_agri_climate_data.csv          # Merged dataset (generated)
│
├── rag_files/
│   ├── rag_corpus.jsonl                      # Text corpus (generated)
│   └── rag_corpus_sample.csv                 # Sample corpus (generated)
│
├── rag_store/
│   └── faiss_index/
│       ├── index.faiss                       # FAISS vector index
│       ├── metadata.pkl                      # Document metadata
│       └── config.json                       # Configuration
│
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))
- **Required datasets** in `data/` folder:
  - `India Agriculture Crop Production.csv`
  - `Sub_Division_IMD_2017.csv`

### Installation

1. **Clone or download the project**
   ```bash
   cd project_samarth
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data and RAG system** (run in order):
   
   **Step 1: Preprocess and merge datasets**
   ```bash
   python data_preprocessing.py
   ```
   This creates `data/merged_agri_climate_data.csv`

   **Step 2: Build RAG corpus**
   ```bash
   python rag_corpus_builder.py
   ```
   This creates `rag_files/rag_corpus.jsonl`

   **Step 3: Create vector store**
   ```bash
   python rag_vector_store.py
   ```
   This creates the FAISS index in `rag_store/faiss_index/`

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Configure and use**
   - Enter your Gemini API key in the sidebar
   - Click "Initialize RAG System"
   - Start asking questions!

---

## 📖 Usage Guide

### Example Queries

1. **Specific Questions**:
   - "What is the rice production in Tamil Nadu in 2015?"
   - "Which state has the highest wheat production in 2020?"

2. **Comparisons**:
   - "Compare rainfall and rice yield in Tamil Nadu for 2015-2020"
   - "Show me crop yield trends for Maharashtra from 2010-2020"

3. **Analysis**:
   - "What is the correlation between rainfall and crop yield in Punjab?"
   - "Which districts in Karnataka produced the most sugarcane in 2020?"

### App Features

- **Ask AI**: Enter your question and get grounded answers
- **Source Documents**: View the retrieved context used for answering
- **Data Visualization**: Automatic charts for trend/comparison queries
- **Data Explorer**: Browse and filter the merged dataset
- **Query History**: View your recent queries

### Sidebar Configuration

- **Gemini API Key**: Enter your API key
- **Model Selection**: Choose between Gemini 2.0 Flash, 1.5 Pro, or 1.5 Flash
- **Top-K Documents**: Number of relevant documents to retrieve (3-15)
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)

---

## 🔧 Technical Details

### RAG Pipeline

1. **Data Preprocessing** (`data_preprocessing.py`):
   - Loads crop production and rainfall datasets
   - Maps states to IMD subdivisions
   - Merges datasets on subdivision and year
   - Outputs: `merged_agri_climate_data.csv`

2. **Corpus Building** (`rag_corpus_builder.py`):
   - Converts each CSV record into a factual sentence
   - Includes metadata (state, district, year, crop, production, yield, rainfall)
   - Outputs: `rag_corpus.jsonl` and sample CSV

3. **Vector Store** (`rag_vector_store.py`):
   - Uses `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
   - Builds FAISS IndexFlatIP for cosine similarity search
   - Stores embeddings and metadata
   - Outputs: FAISS index and metadata pickle files

4. **Q&A System** (`rag_gemini_qa.py`):
   - Retrieves top-k relevant documents using FAISS
   - Builds context string from retrieved docs
   - Calls Google Gemini API with context and query
   - Returns answer with source attribution

5. **Web App** (`app.py`):
   - Streamlit-based UI
   - Interactive query interface
   - Auto-generated Plotly charts
   - Data explorer with filters

### Models & Technologies

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM**: Google Gemini 2.0 Flash / 1.5 Pro
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Web Framework**: Streamlit
- **Visualization**: Plotly

---

## 📊 Data Schema

### Merged Dataset Columns

- `state`: Indian state name
- `district`: District name
- `year_cleaned`: Year (integer)
- `crop`: Crop name
- `production`: Production in tonnes
- `yield`: Yield in tonnes per hectare
- `annual_rainfall`: Annual rainfall in mm

### Corpus Document Format

Each document in `rag_corpus.jsonl`:
```json
{
  "id": "hash_id",
  "text": "In 2015, the district Coimbatore in Tamil Nadu produced 1234.56 tonnes of Rice with yield 2.50 tonnes per hectare and recorded 987.6 mm annual rainfall.",
  "metadata": {
    "state": "Tamil Nadu",
    "district": "Coimbatore",
    "year": 2015,
    "crop": "Rice",
    "production": 1234.56,
    "yield": 2.50,
    "annual_rainfall": 987.6,
    "source": "Ministry of Agriculture (crop) & IMD (rainfall)",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

---

## 🛠️ Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not set"**
   - Set environment variable: `export GEMINI_API_KEY='your-key'`
   - Or enter it in the Streamlit sidebar

2. **"FAISS index not found"**
   - Run `rag_vector_store.py` first
   - Check that `rag_store/faiss_index/` exists

3. **"Merged dataset not found"**
   - Run `data_preprocessing.py` first
   - Check that `data/merged_agri_climate_data.csv` exists

4. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Use Python 3.8+

5. **Slow vector store creation**
   - This is normal for large datasets (can take 10-30 minutes)
   - Progress is shown during embedding generation

---

## 🎨 Customization

### Changing the Embedding Model

Edit `rag_vector_store.py`:
```python
MODEL_NAME = "all-mpnet-base-v2"  # Better but slower
EMBEDDING_DIM = 768  # Update dimension
```

### Changing Gemini Model

In `app.py` or `rag_gemini_qa.py`, change:
```python
gemini_model = "gemini-1.5-pro"  # More powerful but slower
```

### Adjusting Retrieval

In `app.py` sidebar, adjust:
- **Top-K**: More documents = more context but slower
- **Temperature**: Lower = more focused, Higher = more creative

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **Data Sources**:
  - Ministry of Agriculture (crop production data)
  - Indian Meteorological Department (IMD) - rainfall data
- **Technologies**:
  - Google Gemini API
  - FAISS (Facebook AI)
  - Sentence Transformers
  - Streamlit

---

## 📧 Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all setup steps were completed

---

## 🔮 Future Enhancements

- [ ] Add more data sources (temperature, humidity, soil data)
- [ ] Implement query caching with Redis
- [ ] Add export functionality (PDF reports)
- [ ] Multi-language support
- [ ] Real-time data updates
- [ ] Advanced analytics dashboard
- [ ] User authentication
- [ ] Batch query processing

---

**Built with ❤️ for Project Samarth**

*Last updated: 2024*

