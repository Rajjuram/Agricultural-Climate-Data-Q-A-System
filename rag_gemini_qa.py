"""
RAG-powered Q&A System using Google Gemini
Retrieves relevant context from FAISS and generates answers using Gemini LLM
"""

import json
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Optional
import os

# Optional Gemini import - will use fallback if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

class RAGGeminiQA:
    """RAG-powered Q&A system using Gemini LLM"""
    
    def __init__(
        self,
        faiss_index_dir: Path = Path("rag_store/faiss_index"),
        model_name: str = "all-MiniLM-L6-v2",
        gemini_model: str = "gemini-2.0-flash-exp",
        top_k: int = 5,
        temperature: float = 0.7
    ):
        """
        Initialize RAG Q&A system
        
        Args:
            faiss_index_dir: Path to FAISS index directory
            model_name: Sentence transformer model name
            gemini_model: Gemini model name (gemini-2.0-flash-exp or gemini-1.5-pro)
            top_k: Number of top documents to retrieve
            temperature: Temperature for Gemini generation
        """
        self.faiss_index_dir = Path(faiss_index_dir)
        self.top_k = top_k
        self.temperature = temperature
        self.gemini_model_name = gemini_model
        
        print(f"🔍 Initializing RAG Q&A System...")
        print(f"  FAISS index dir: {self.faiss_index_dir}")
        
        # Load configuration
        config_file = self.faiss_index_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}. Run rag_vector_store.py first.")
        
        with open(config_file, "r") as f:
            self.config = json.load(f)
        
        # Load FAISS index
        index_file = self.faiss_index_dir / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}. Run rag_vector_store.py first.")
        
        print(f"  Loading FAISS index...")
        self.index = faiss.read_index(str(index_file))
        print(f"  ✅ Loaded index with {self.index.ntotal:,} vectors")
        
        # Load metadata
        metadata_file = self.faiss_index_dir / "metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, "rb") as f:
            self.documents = pickle.load(f)
        print(f"  ✅ Loaded {len(self.documents):,} document metadata")
        
        # Load embedding model
        print(f"  Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"  ✅ Embedding model loaded")
        
        # Initialize Gemini (optional - will use fallback if not available)
        self.gemini_model = None
        self.use_gemini = False
        
        if not GEMINI_AVAILABLE:
            print(f"  ⚠️ google-generativeai package not installed - using local answer generation")
            self.use_gemini = False
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    print(f"  Initializing Gemini API...")
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel(gemini_model)
                    self.use_gemini = True
                    print(f"  ✅ Gemini model '{gemini_model}' initialized")
                except Exception as e:
                    print(f"  ⚠️ Gemini initialization failed: {str(e)}")
                    print(f"  📝 Will use local answer generation from vector database")
                    self.use_gemini = False
            else:
                print(f"  ⚠️ GEMINI_API_KEY not set - using local answer generation")
                self.use_gemini = False
        
        print(f"✅ RAG Q&A System ready! (Mode: {'Gemini API' if self.use_gemini else 'Local Vector DB Only'})\n")
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve top-k relevant documents from FAISS
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve (defaults to self.top_k)
        
        Returns:
            List of document dictionaries with text and metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve documents
        retrieved_docs = []
        for idx, dist in zip(indices[0], distances[0]):
            doc = self.documents[idx].copy()
            doc["similarity_score"] = float(dist)
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def build_context_string(self, retrieved_docs: List[Dict]) -> str:
        """
        Build context string from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
        
        Returns:
            Formatted context string
        """
        context_parts = []
        context_parts.append("=== RELEVANT DATA CONTEXT ===\n")
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Text: {doc['text']}")
            metadata = doc.get('metadata', {})
            if metadata:
                meta_info = []
                if metadata.get('state'):
                    meta_info.append(f"State: {metadata['state']}")
                if metadata.get('district'):
                    meta_info.append(f"District: {metadata['district']}")
                if metadata.get('year'):
                    meta_info.append(f"Year: {metadata['year']}")
                if metadata.get('crop'):
                    meta_info.append(f"Crop: {metadata['crop']}")
                if meta_info:
                    context_parts.append(f"Details: {', '.join(meta_info)}")
            context_parts.append("")  # Empty line
        
        context_parts.append("=== END CONTEXT ===\n")
        return "\n".join(context_parts)
    
    def generate_answer_from_context(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate answer directly from retrieved documents without Gemini API
        Creates intelligent summaries and analysis from the data
        
        Args:
            query: User query string
            retrieved_docs: List of retrieved document dictionaries
        
        Returns:
            Human-readable answer string
        """
        if not retrieved_docs:
            return "I couldn't find any relevant data to answer your question. Please try rephrasing or asking about a different topic."
        
        query_lower = query.lower()
        
        # Extract structured data from documents
        data_points = []
        for doc in retrieved_docs:
            meta = doc.get('metadata', {})
            if meta:
                data_points.append({
                    'state': meta.get('state', 'Unknown'),
                    'district': meta.get('district', 'Unknown'),
                    'year': meta.get('year'),
                    'crop': meta.get('crop', 'Unknown'),
                    'production': meta.get('production'),
                    'yield': meta.get('yield'),
                    'rainfall': meta.get('annual_rainfall'),
                    'text': doc.get('text', '')
                })
        
        # Analyze query type and generate appropriate answer
        answer_parts = []
        
        # Specific value questions (what, how much, how many)
        if any(word in query_lower for word in ['what is', 'how much', 'how many', 'what was', 'tell me']):
            if 'compare' in query_lower or 'comparison' in query_lower:
                # Comparison query
                answer_parts.append(self._generate_comparison_answer(query, data_points))
            elif 'trend' in query_lower or 'over time' in query_lower or any(str(y) in query for y in range(1997, 2018)):
                # Trend query
                answer_parts.append(self._generate_trend_answer(query, data_points))
            else:
                # Specific value query
                answer_parts.append(self._generate_specific_answer(query, data_points))
        
        # Comparison questions
        elif 'compare' in query_lower or 'comparison' in query_lower:
            answer_parts.append(self._generate_comparison_answer(query, data_points))
        
        # Trend/time series questions
        elif any(word in query_lower for word in ['trend', 'over time', 'year', '2015', '2016', '2017', '2018', '2019', '2020']):
            answer_parts.append(self._generate_trend_answer(query, data_points))
        
        # Highest/lowest questions
        elif any(word in query_lower for word in ['highest', 'lowest', 'maximum', 'minimum', 'top', 'best', 'worst']):
            answer_parts.append(self._generate_extreme_answer(query, data_points))
        
        # Correlation/relationship questions
        elif any(word in query_lower for word in ['correlation', 'relationship', 'relation', 'impact', 'effect']):
            answer_parts.append(self._generate_relationship_answer(query, data_points))
        
        # Default: general summary
        else:
            answer_parts.append(self._generate_summary_answer(query, data_points))
        
        # Add data source note
        answer_parts.append(f"\n\n*Based on {len(retrieved_docs)} relevant records from the agricultural and climate database.*")
        
        return "\n".join(answer_parts)
    
    def _generate_specific_answer(self, query: str, data_points: List[Dict]) -> str:
        """Generate answer for specific value questions"""
        if not data_points:
            return "I couldn't find relevant data for your query."
        
        # Filter data based on query
        filtered = self._filter_data_by_query(query, data_points)
        
        if not filtered:
            return f"Based on the available data, I couldn't find specific information matching your query: '{query}'. Please try rephrasing or specifying a different time period, state, or crop."
        
        answer_parts = []
        
        # Group by relevant dimensions
        grouped = {}
        for dp in filtered[:20]:  # Limit to top 20
            key = f"{dp.get('state', '')}_{dp.get('crop', '')}_{dp.get('year', '')}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(dp)
        
        for key, items in list(grouped.items())[:5]:  # Show top 5 groups
            item = items[0]
            parts = []
            
            if item.get('state'):
                parts.append(f"**{item['state']}**")
            if item.get('district') and item['district'] != 'Unknown':
                parts.append(f"(District: {item['district']})")
            if item.get('year'):
                parts.append(f"- Year {item['year']}")
            if item.get('crop'):
                parts.append(f"- Crop: {item['crop']}")
            
            answer_parts.append(" | ".join(parts))
            
            details = []
            if item.get('production') is not None:
                details.append(f"Production: **{item['production']:,.2f} tonnes**")
            if item.get('yield') is not None:
                details.append(f"Yield: **{item['yield']:.2f} tonnes/hectare**")
            if item.get('rainfall') is not None:
                details.append(f"Annual Rainfall: **{item['rainfall']:.1f} mm**")
            
            if details:
                answer_parts.append("  " + " | ".join(details))
            answer_parts.append("")
        
        return "\n".join(answer_parts)
    
    def _generate_comparison_answer(self, query: str, data_points: List[Dict]) -> str:
        """Generate answer for comparison questions"""
        if not data_points:
            return "I couldn't find relevant data for comparison."
        
        filtered = self._filter_data_by_query(query, data_points)
        
        if len(filtered) < 2:
            return "I found limited data for comparison. Please ensure your query includes multiple states, years, or crops to compare."
        
        answer_parts = []
        answer_parts.append("**Comparison Analysis:**\n")
        
        # Group by relevant dimensions
        by_state = {}
        by_year = {}
        by_crop = {}
        
        for dp in filtered[:50]:
            if dp.get('state'):
                if dp['state'] not in by_state:
                    by_state[dp['state']] = []
                by_state[dp['state']].append(dp)
            
            if dp.get('year'):
                if dp['year'] not in by_year:
                    by_year[dp['year']] = []
                by_year[dp['year']].append(dp)
            
            if dp.get('crop'):
                if dp['crop'] not in by_crop:
                    by_crop[dp['crop']] = []
                by_crop[dp['crop']].append(dp)
        
        # Compare by year (trends)
        if 'year' in query.lower() or len(by_year) > 1:
            answer_parts.append("*Year-wise Comparison:*")
            for year in sorted(by_year.keys())[-10:]:  # Last 10 years
                items = by_year[year]
                prods = [x['production'] for x in items if x.get('production') is not None]
                ylds = [x['yield'] for x in items if x.get('yield') is not None]
                rains = [x['rainfall'] for x in items if x.get('rainfall') is not None]
                
                avg_prod = np.mean(prods) if prods else 0
                avg_yield = np.mean(ylds) if ylds else 0
                avg_rain = np.mean(rains) if rains else 0
                
                answer_parts.append(f"  **{year}**: Avg Production: {avg_prod:,.1f} tonnes | Avg Yield: {avg_yield:.2f} t/ha | Avg Rainfall: {avg_rain:.1f} mm")
        
        # Compare by state
        elif len(by_state) > 1:
            answer_parts.append("*State-wise Comparison:*")
            for state in sorted(by_state.keys())[:10]:  # Top 10 states
                items = by_state[state]
                prods = [x['production'] for x in items if x.get('production') is not None]
                ylds = [x['yield'] for x in items if x.get('yield') is not None]
                
                avg_prod = np.mean(prods) if prods else 0
                avg_yield = np.mean(ylds) if ylds else 0
                
                answer_parts.append(f"  **{state}**: Avg Production: {avg_prod:,.1f} tonnes | Avg Yield: {avg_yield:.2f} t/ha")
        
        # Compare by crop
        elif len(by_crop) > 1:
            answer_parts.append("*Crop-wise Comparison:*")
            for crop in sorted(by_crop.keys())[:10]:
                items = by_crop[crop]
                prods = [x['production'] for x in items if x.get('production') is not None]
                ylds = [x['yield'] for x in items if x.get('yield') is not None]
                
                avg_prod = np.mean(prods) if prods else 0
                avg_yield = np.mean(ylds) if ylds else 0
                
                answer_parts.append(f"  **{crop}**: Avg Production: {avg_prod:,.1f} tonnes | Avg Yield: {avg_yield:.2f} t/ha")
        
        return "\n".join(answer_parts)
    
    def _generate_trend_answer(self, query: str, data_points: List[Dict]) -> str:
        """Generate answer for trend/time series questions"""
        if not data_points:
            return "I couldn't find relevant data for trend analysis."
        
        filtered = self._filter_data_by_query(query, data_points)
        
        if len(filtered) < 2:
            return "I found limited data for trend analysis. Please ensure your query covers multiple years."
        
        answer_parts = []
        answer_parts.append("**Trend Analysis:**\n")
        
        # Group by year
        by_year = {}
        for dp in filtered:
            year = dp.get('year')
            if year:
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(dp)
        
        if not by_year:
            return self._generate_summary_answer(query, data_points)
        
        # Calculate trends
        years = sorted(by_year.keys())
        if len(years) < 2:
            return self._generate_summary_answer(query, data_points)
        
        answer_parts.append("*Year-wise Data:*")
        
        productions = []
        yields = []
        rainfalls = []
        
        for year in years[-10:]:  # Last 10 years
            items = by_year[year]
            prods = [x['production'] for x in items if x.get('production') is not None]
            ylds = [x['yield'] for x in items if x.get('yield') is not None]
            rains = [x['rainfall'] for x in items if x.get('rainfall') is not None]
            
            avg_prod = np.mean(prods) if prods else None
            avg_yield = np.mean(ylds) if ylds else None
            avg_rain = np.mean(rains) if rains else None
            
            if avg_prod: productions.append((year, avg_prod))
            if avg_yield: yields.append((year, avg_yield))
            if avg_rain: rainfalls.append((year, avg_rain))
            
            parts = [f"**{year}**"]
            if avg_prod: parts.append(f"Production: {avg_prod:,.1f} tonnes")
            if avg_yield: parts.append(f"Yield: {avg_yield:.2f} t/ha")
            if avg_rain: parts.append(f"Rainfall: {avg_rain:.1f} mm")
            answer_parts.append("  " + " | ".join(parts))
        
        # Calculate trend direction
        if len(productions) >= 2 and productions[0][1] > 0:
            trend = "increasing" if productions[-1][1] > productions[0][1] else "decreasing"
            try:
                change_pct = ((productions[-1][1] - productions[0][1]) / productions[0][1]) * 100
                answer_parts.append(f"\n*Production Trend: {trend} by {abs(change_pct):.1f}% from {productions[0][0]} to {productions[-1][0]}*")
            except (ZeroDivisionError, ValueError):
                answer_parts.append(f"\n*Production Trend: {trend} from {productions[0][0]} to {productions[-1][0]}*")
        
        return "\n".join(answer_parts)
    
    def _generate_extreme_answer(self, query: str, data_points: List[Dict]) -> str:
        """Generate answer for highest/lowest questions"""
        if not data_points:
            return "I couldn't find relevant data."
        
        filtered = self._filter_data_by_query(query, data_points)
        
        if not filtered:
            return "I couldn't find data matching your criteria."
        
        query_lower = query.lower()
        
        answer_parts = []
        
        if 'highest' in query_lower or 'maximum' in query_lower or 'top' in query_lower:
            # Find maximum production
            max_prod = max([x for x in filtered if x.get('production') is not None], 
                          key=lambda x: x.get('production', 0), default=None)
            if max_prod:
                answer_parts.append(f"**Highest Production:**")
                answer_parts.append(f"  State: {max_prod.get('state', 'Unknown')}")
                answer_parts.append(f"  District: {max_prod.get('district', 'Unknown')}")
                answer_parts.append(f"  Crop: {max_prod.get('crop', 'Unknown')}")
                answer_parts.append(f"  Year: {max_prod.get('year', 'Unknown')}")
                answer_parts.append(f"  Production: **{max_prod['production']:,.2f} tonnes**")
                if max_prod.get('yield'):
                    answer_parts.append(f"  Yield: {max_prod['yield']:.2f} tonnes/hectare")
                if max_prod.get('rainfall'):
                    answer_parts.append(f"  Rainfall: {max_prod['rainfall']:.1f} mm")
        
        elif 'lowest' in query_lower or 'minimum' in query_lower:
            # Find minimum production (excluding zeros)
            min_prod = min([x for x in filtered if x.get('production') and x['production'] > 0], 
                          key=lambda x: x.get('production', float('inf')), default=None)
            if min_prod:
                answer_parts.append(f"**Lowest Production:**")
                answer_parts.append(f"  State: {min_prod.get('state', 'Unknown')}")
                answer_parts.append(f"  District: {min_prod.get('district', 'Unknown')}")
                answer_parts.append(f"  Crop: {min_prod.get('crop', 'Unknown')}")
                answer_parts.append(f"  Year: {min_prod.get('year', 'Unknown')}")
                answer_parts.append(f"  Production: **{min_prod['production']:,.2f} tonnes**")
        
        return "\n".join(answer_parts) if answer_parts else "Could not determine extreme values."
    
    def _generate_relationship_answer(self, query: str, data_points: List[Dict]) -> str:
        """Generate answer for correlation/relationship questions"""
        if not data_points:
            return "I couldn't find relevant data for relationship analysis."
        
        filtered = [dp for dp in data_points if dp.get('rainfall') is not None and dp.get('yield') is not None]
        
        if len(filtered) < 10:
            return "I found limited data for relationship analysis. Need more records with both rainfall and yield data."
        
        answer_parts = []
        answer_parts.append("**Rainfall vs Crop Yield Relationship Analysis:**\n")
        
        # Calculate correlation
        rainfalls = [dp['rainfall'] for dp in filtered]
        yields = [dp['yield'] for dp in filtered]
        
        if len(rainfalls) > 1 and len(yields) > 1 and len(rainfalls) == len(yields):
            try:
                correlation = np.corrcoef(rainfalls, yields)[0, 1]
                if not np.isnan(correlation):
                    answer_parts.append(f"*Correlation Coefficient: {correlation:.3f}*")
                    
                    if correlation > 0.3:
                        answer_parts.append("  - **Positive relationship**: Higher rainfall tends to correlate with higher crop yields.")
                    elif correlation < -0.3:
                        answer_parts.append("  - **Negative relationship**: Higher rainfall tends to correlate with lower crop yields.")
                    else:
                        answer_parts.append("  - **Weak relationship**: Rainfall and crop yield show limited correlation in the data.")
            except Exception as e:
                answer_parts.append("*Correlation calculation unavailable due to insufficient data variance.*")
        
        # Average values
        if rainfalls and yields:
            avg_rain = np.mean(rainfalls)
            avg_yield = np.mean(yields)
            answer_parts.append(f"\n*Average Values:*")
            answer_parts.append(f"  - Average Rainfall: {avg_rain:.1f} mm")
            answer_parts.append(f"  - Average Yield: {avg_yield:.2f} tonnes/hectare")
        
        return "\n".join(answer_parts)
    
    def _generate_summary_answer(self, query: str, data_points: List[Dict]) -> str:
        """Generate general summary answer"""
        if not data_points:
            return "I couldn't find relevant data for your query."
        
        filtered = self._filter_data_by_query(query, data_points)
        
        if not filtered:
            return f"Based on the available data, I couldn't find specific information matching your query. Please try rephrasing or specifying different parameters."
        
        answer_parts = []
        answer_parts.append("**Summary of Available Data:**\n")
        
        # Aggregate statistics
        prods = [x['production'] for x in filtered if x.get('production') is not None]
        ylds = [x['yield'] for x in filtered if x.get('yield') is not None]
        rains = [x['rainfall'] for x in filtered if x.get('rainfall') is not None]
        
        if prods:
            answer_parts.append(f"*Production Statistics:*")
            answer_parts.append(f"  - Average: {np.mean(prods):,.1f} tonnes")
            answer_parts.append(f"  - Maximum: {np.max(prods):,.1f} tonnes")
            answer_parts.append(f"  - Minimum: {np.min(prods):,.1f} tonnes")
        
        if ylds:
            answer_parts.append(f"\n*Yield Statistics:*")
            answer_parts.append(f"  - Average: {np.mean(ylds):.2f} tonnes/hectare")
            answer_parts.append(f"  - Maximum: {np.max(ylds):.2f} tonnes/hectare")
            answer_parts.append(f"  - Minimum: {np.min(ylds):.2f} tonnes/hectare")
        
        if rains:
            answer_parts.append(f"\n*Rainfall Statistics:*")
            answer_parts.append(f"  - Average: {np.mean(rains):.1f} mm")
            answer_parts.append(f"  - Maximum: {np.max(rains):.1f} mm")
            answer_parts.append(f"  - Minimum: {np.min(rains):.1f} mm")
        
        # Sample records
        answer_parts.append(f"\n*Sample Records ({min(5, len(filtered))} of {len(filtered)}):*")
        for dp in filtered[:5]:
            parts = []
            if dp.get('state'): parts.append(dp['state'])
            if dp.get('year'): parts.append(f"({dp['year']})")
            if dp.get('crop'): parts.append(f"- {dp['crop']}")
            if parts:
                answer_parts.append("  - " + " ".join(parts))
        
        return "\n".join(answer_parts)
    
    def _filter_data_by_query(self, query: str, data_points: List[Dict]) -> List[Dict]:
        """Filter data points based on query keywords"""
        query_lower = query.lower()
        filtered = []
        
        for dp in data_points:
            match = True
            
            # Filter by state
            states = ['tamil nadu', 'maharashtra', 'punjab', 'karnataka', 'gujarat', 'west bengal', 
                     'bihar', 'andhra pradesh', 'telangana', 'rajasthan', 'uttar pradesh']
            state_in_query = None
            for state in states:
                if state in query_lower:
                    state_in_query = state.title()
                    break
            
            if state_in_query and dp.get('state', '').lower() != state_in_query.lower():
                continue
            
            # Filter by crop
            crops = ['rice', 'wheat', 'sugarcane', 'cotton', 'maize', 'banana', 'potato', 'tomato']
            crop_in_query = None
            for crop in crops:
                if crop in query_lower:
                    crop_in_query = crop.title()
                    break
            
            if crop_in_query and dp.get('crop', '').lower() != crop_in_query.lower():
                continue
            
            # Filter by year
            years_in_query = [str(y) for y in range(1997, 2018) if str(y) in query]
            if years_in_query:
                query_years = [int(y) for y in years_in_query]
                if dp.get('year') and dp['year'] not in query_years:
                    # Allow if year is within range if range specified
                    if 'to' in query_lower or '-' in query_lower:
                        # Try to extract year range
                        pass  # Keep for now
                    else:
                        continue
            
            filtered.append(dp)
        
        return filtered if filtered else data_points[:20]  # Return filtered or top 20
    
    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Answer a query using RAG pipeline
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            temperature: Generation temperature
            include_sources: Whether to include source documents in response
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if top_k is None:
            top_k = self.top_k
        if temperature is None:
            temperature = self.temperature
        
        print(f"🔍 Processing query: '{query}'")
        
        # Step 1: Retrieve relevant context
        print(f"  Step 1: Retrieving top-{top_k} relevant documents...")
        retrieved_docs = self.retrieve_context(query, top_k)
        print(f"  ✅ Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Build context
        context = self.build_context_string(retrieved_docs)
        
        # Step 3: Build prompt for Gemini
        prompt = f"""You are an expert AI assistant specializing in Indian agriculture, climate, and crop production data analysis.

You have access to factual data from the Ministry of Agriculture (crop production) and IMD (Indian Meteorological Department - rainfall data).

Use the following context to answer the user's question accurately and comprehensively. If the question asks for comparisons, trends, or analysis, provide detailed insights with specific numbers and facts from the data.

CONTEXT DATA:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context data above.
2. If the question asks for comparisons, charts, or trends, mention the specific states, districts, years, crops, rainfall values, yields, and production figures.
3. If specific data is not available in the context, clearly state what information is missing.
4. Be precise with numbers, units (tonnes, mm, hectares), and years.
5. If asked for analysis, provide insights about relationships between rainfall and crop yield/production.
6. Write in a clear, professional, and helpful tone.

ANSWER:"""
        
        # Step 4: Generate answer - use Gemini if available, otherwise use local generation
        if self.use_gemini and self.gemini_model and GEMINI_AVAILABLE and genai is not None:
            print(f"  Step 2: Generating answer with Gemini ({self.gemini_model_name})...")
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                )
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                answer = response.text.strip()
                print(f"  ✅ Answer generated with Gemini ({len(answer)} characters)")
                
            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                print(f"  ⚠️ Gemini API error: {error_msg}")
                print(f"  📝 Falling back to local answer generation from vector database...")
                # Fallback to local generation
                answer = self.generate_answer_from_context(query, retrieved_docs)
                print(f"  ✅ Answer generated from vector database ({len(answer)} characters)")
        else:
            # Use local generation directly
            print(f"  Step 2: Generating answer from vector database (no Gemini API)...")
            answer = self.generate_answer_from_context(query, retrieved_docs)
            print(f"  ✅ Answer generated from vector database ({len(answer)} characters)")
        
        # Step 5: Prepare response
        result = {
            "query": query,
            "answer": answer,
            "num_sources": len(retrieved_docs),
            "temperature": temperature
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "similarity_score": doc.get("similarity_score", 0.0)
                }
                for doc in retrieved_docs
            ]
        
        return result


def answer_query(query: str, **kwargs) -> Dict:
    """
    Convenience function to answer a query
    
    Args:
        query: User query string
        **kwargs: Additional arguments for RAGGeminiQA.answer_query()
    
    Returns:
        Dictionary with answer and sources
    """
    # Initialize system (could be cached in production)
    qa_system = RAGGeminiQA()
    return qa_system.answer_query(query, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("🧪 Testing RAG Q&A System")
    print("=" * 60)
    
    try:
        qa = RAGGeminiQA()
        
        test_queries = [
            "What is the rice production in Tamil Nadu in 2015?",
            "Compare rainfall and rice yield in Tamil Nadu for 2015-2020",
            "Which state has the highest wheat production in 2020?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            result = qa.answer_query(query, include_sources=True)
            
            print(f"\n📝 Answer:")
            print(result["answer"])
            print(f"\n📚 Sources ({result['num_sources']} documents):")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n  [{i}] {source['text'][:150]}...")
                print(f"      Similarity: {source['similarity_score']:.3f}")
            
            print("\n")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

