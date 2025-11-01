"""
Streamlit Web App for Project Samarth - AI Q&A System
RAG-powered agricultural and climate data analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from rag_gemini_qa import RAGGeminiQA
import json
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Project Samarth - AI Q&A",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #fff;
        padding: 1rem;
        border-radius: 0.3rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "answers_cache" not in st.session_state:
    st.session_state.answers_cache = {}
if "example_query" not in st.session_state:
    st.session_state.example_query = ""

# Title and header
st.markdown('<p class="main-header">🌾 Project Samarth</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Agricultural & Climate Data Q&A System</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Gemini API Key input
    st.subheader("🔑 API Configuration")
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Enter your Google Gemini API key. Get one from https://makersuite.google.com/app/apikey"
    )
    
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    
    # Model selection
    st.subheader("🤖 Model Settings")
    gemini_model = st.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
        help="Select the Gemini model to use"
    )
    
    top_k = st.slider(
        "Top-K Documents",
        min_value=3,
        max_value=15,
        value=5,
        help="Number of relevant documents to retrieve for context"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in model output (0 = deterministic, 1 = creative)"
    )
    
    # Initialize button
    if st.button("🔄 Initialize RAG System", use_container_width=True):
        with st.spinner("Initializing RAG system... This may take a moment."):
            try:
                st.session_state.qa_system = RAGGeminiQA(
                    gemini_model=gemini_model,
                    top_k=top_k,
                    temperature=temperature
                )
                if st.session_state.qa_system.use_gemini:
                    st.success("✅ RAG system initialized successfully with Gemini API!")
                else:
                    st.success("✅ RAG system initialized successfully (using local vector database only)")
                    st.info("💡 Note: Gemini API not available. Answers will be generated from the vector database.")
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                st.info("Make sure you have run the setup scripts:\n1. data_preprocessing.py\n2. rag_corpus_builder.py\n3. rag_vector_store.py")
    
    st.divider()
    
    # Query examples
    st.subheader("💡 Example Queries")
    example_queries = [
        "What is the rice production in Tamil Nadu in 2015?",
        "Compare rainfall and rice yield in Tamil Nadu for 2015-2020",
        "Which state has the highest wheat production?",
        "Show me crop yield trends for Maharashtra from 2010-2020",
        "What is the correlation between rainfall and crop yield in Punjab?",
        "Which districts in Karnataka produced the most sugarcane in 2020?"
    ]
    
    for i, example in enumerate(example_queries):
        if st.button(f"📝 {example[:50]}...", key=f"example_{i}", use_container_width=True):
            st.session_state.example_query = example
            st.rerun()

# Main content area
if st.session_state.qa_system is None:
    st.info("👆 Please configure and initialize the RAG system using the sidebar.")
    
    # Show setup instructions
    with st.expander("📋 Setup Instructions", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Prepare Data** (if not done):
           ```bash
           python data_preprocessing.py
           ```
        
        2. **Build RAG Corpus**:
           ```bash
           python rag_corpus_builder.py
           ```
        
        3. **Create Vector Store**:
           ```bash
           python rag_vector_store.py
           ```
        
        4. **Configure & Run App**:
           - Enter your Gemini API key in the sidebar
           - Click "Initialize RAG System"
           - Start asking questions!
        """)
else:
    # Main Q&A interface
    st.header("💬 Ask Your Question")
    
    # Query input
    query_text = st.text_area(
        "Enter your question about Indian agriculture, rainfall, or crop production:",
        value=st.session_state.get("example_query", ""),
        height=100,
        placeholder="e.g., Compare rainfall and rice yield in Tamil Nadu for 2015-2020"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        ask_button = st.button("🔍 Ask AI", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.example_query = ""
        st.rerun()
    
    # Process query
    if ask_button and query_text.strip():
        with st.spinner("🤔 Thinking... Generating answer based on retrieved data..."):
            try:
                # Check cache
                cache_key = f"{query_text}_{top_k}_{temperature}"
                if cache_key in st.session_state.answers_cache:
                    result = st.session_state.answers_cache[cache_key]
                    st.info("📦 Answer from cache")
                else:
                    # Get answer from RAG system
                    result = st.session_state.qa_system.answer_query(
                        query=query_text,
                        top_k=top_k,
                        temperature=temperature,
                        include_sources=True
                    )
                    # Cache result
                    st.session_state.answers_cache[cache_key] = result
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "query": query_text,
                        "timestamp": datetime.now().isoformat(),
                        "answer_preview": result["answer"][:100] + "..."
                    })
                    
                    # Clear example query after use
                    st.session_state.example_query = ""
                
                # Display answer
                st.markdown("---")
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.subheader("📝 Answer")
                st.markdown(result["answer"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display sources
                with st.expander(f"📚 Source Documents ({result['num_sources']} documents)", expanded=False):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Source {i}** (Similarity: {source['similarity_score']:.3f})")
                        st.markdown(f"*{source['text']}*")
                        
                        metadata = source.get("metadata", {})
                        if metadata:
                            meta_info = []
                            if metadata.get("state"):
                                meta_info.append(f"State: {metadata['state']}")
                            if metadata.get("district"):
                                meta_info.append(f"District: {metadata['district']}")
                            if metadata.get("year"):
                                meta_info.append(f"Year: {metadata['year']}")
                            if metadata.get("crop"):
                                meta_info.append(f"Crop: {metadata['crop']}")
                            if meta_info:
                                st.caption(" | ".join(meta_info))
                        st.divider()
                
                # Auto-generate charts for certain query types
                if any(keyword in query_text.lower() for keyword in ["compare", "trend", "show", "plot", "chart", "graph"]):
                    st.subheader("📊 Data Visualization")
                    try:
                        # Extract data from sources for visualization
                        chart_data = []
                        for source in result["sources"]:
                            meta = source.get("metadata", {})
                            if meta.get("year") and meta.get("crop"):
                                chart_data.append({
                                    "Year": meta.get("year"),
                                    "Crop": meta.get("crop"),
                                    "State": meta.get("state", "Unknown"),
                                    "Production": meta.get("production", 0),
                                    "Yield": meta.get("yield", 0),
                                    "Rainfall": meta.get("annual_rainfall", 0)
                                })
                        
                        if chart_data:
                            df_chart = pd.DataFrame(chart_data)
                            
                            tab1, tab2, tab3 = st.tabs(["Production Trends", "Yield Trends", "Rainfall vs Yield"])
                            
                            with tab1:
                                if "Production" in df_chart.columns:
                                    fig_prod = px.line(
                                        df_chart.groupby(["Year", "State", "Crop"])["Production"].mean().reset_index(),
                                        x="Year",
                                        y="Production",
                                        color="Crop",
                                        title="Production Trends Over Time",
                                        labels={"Production": "Production (tonnes)", "Year": "Year"}
                                    )
                                    st.plotly_chart(fig_prod, use_container_width=True)
                            
                            with tab2:
                                if "Yield" in df_chart.columns:
                                    fig_yield = px.line(
                                        df_chart.groupby(["Year", "State", "Crop"])["Yield"].mean().reset_index(),
                                        x="Year",
                                        y="Yield",
                                        color="Crop",
                                        title="Yield Trends Over Time",
                                        labels={"Yield": "Yield (tonnes/hectare)", "Year": "Year"}
                                    )
                                    st.plotly_chart(fig_yield, use_container_width=True)
                            
                            with tab3:
                                if "Rainfall" in df_chart.columns and "Yield" in df_chart.columns:
                                    fig_scatter = px.scatter(
                                        df_chart,
                                        x="Rainfall",
                                        y="Yield",
                                        color="Crop",
                                        size="Production",
                                        hover_data=["State", "Year"],
                                        title="Rainfall vs Crop Yield",
                                        labels={"Rainfall": "Annual Rainfall (mm)", "Yield": "Yield (tonnes/hectare)"}
                                    )
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate charts: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.exception(e)
    
    # Data Explorer tab
    st.divider()
    
    with st.expander("📊 Data Explorer", expanded=False):
        st.subheader("Explore Merged Agricultural & Climate Data")
        
        # Load and display merged data
        merged_file = Path("data") / "merged_agri_climate_data.csv"
        if merged_file.exists():
            try:
                df = pd.read_csv(merged_file)
                st.success(f"✅ Loaded {len(df):,} records")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    states = st.multiselect("Filter by State", options=sorted(df["state"].unique()) if "state" in df.columns else [])
                with col2:
                    crops = st.multiselect("Filter by Crop", options=sorted(df["crop"].unique()) if "crop" in df.columns else [])
                with col3:
                    years = st.multiselect("Filter by Year", options=sorted(df["year_cleaned"].unique()) if "year_cleaned" in df.columns else [])
                
                # Apply filters
                filtered_df = df.copy()
                if states:
                    filtered_df = filtered_df[filtered_df["state"].isin(states)]
                if crops:
                    filtered_df = filtered_df[filtered_df["crop"].isin(crops)]
                if years:
                    filtered_df = filtered_df[filtered_df["year_cleaned"].isin(years)]
                
                st.dataframe(filtered_df.head(1000), use_container_width=True)
                
                # Summary statistics
                if not filtered_df.empty:
                    st.subheader("📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(filtered_df))
                    with col2:
                        if "production" in filtered_df.columns:
                            st.metric("Avg Production", f"{filtered_df['production'].mean():.2f} tonnes")
                    with col3:
                        if "yield" in filtered_df.columns:
                            st.metric("Avg Yield", f"{filtered_df['yield'].mean():.2f} t/ha")
                    with col4:
                        if "annual_rainfall" in filtered_df.columns:
                            st.metric("Avg Rainfall", f"{filtered_df['annual_rainfall'].mean():.1f} mm")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        else:
            st.warning(f"⚠️ Merged data file not found: {merged_file}")
            st.info("Please run data_preprocessing.py first")
    
    # Query History
    if st.session_state.query_history:
        with st.expander("📜 Query History", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                st.markdown(f"**{i}. {entry['query']}**")
                st.caption(f"Preview: {entry['answer_preview']}")
                st.caption(f"Time: {entry['timestamp']}")
                st.divider()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🌾 Project Samarth - RAG-powered AI Q&A System for Indian Agriculture & Climate Data"
    "<br>Powered by Google Gemini & FAISS Vector Search"
    "</div>",
    unsafe_allow_html=True
)

