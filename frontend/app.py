import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Self-Healing RAG System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Self-Healing RAG Engine")
st.caption("Closed-Loop Retrieval Pipeline powered by LangChain, Groq, ChromaDB, & Cross-Encoder Reranking")

# --- Sidebar for System Status ---
with st.sidebar:
    st.header("⚙️ Backend Config")
    api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000")
    
    st.divider()
    
    if st.button("Check API Health"):
        try:
            res = requests.get(f"{api_url}/health", timeout=3)
            if res.status_code == 200:
                st.success("API Server Online 🟢")
            else:
                st.error("API Unreachable 🔴")
        except Exception as e:
            st.error(f"Connection Error: {e}")

    st.markdown("### 🛠️ Active Pipeline Layers")
    st.markdown("""
    * **1. HyDE:** Generates synthetic research context.
    * **2. Vector DB:** BAAI BGE-Small embeddings.
    * **3. CRAG:** Evaluates document relevance.
    * **4. Cross-Encoder:** Precision reranking.
    * **5. Groq:** Llama 3 70B synthesis.
    """)

# --- Main Query Interface ---
query = st.text_area(
    "Ask a research question based on your ingested ArXiv papers:",
    placeholder="e.g., What is the FederatedSGD algorithm?",
    height=100
)

if st.button("Run Pipeline", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid query before submitting.")
    else:
        with st.spinner("Executing Self-Healing Pipeline (HyDE ➔ CRAG ➔ Rerank ➔ Synthesis)..."):
            try:
                response = requests.post(
                    f"{api_url}/api/chat",
                    json={"query": query},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    diagnostics = data.get("diagnostics", {})
                    
                    # --- Render Generated Answer ---
                    st.markdown("### 💡 Generated Answer")
                    st.success(answer)
                    
                    st.divider()
                    
                    # --- Pipeline Telemetry Metrics ---
                    st.markdown("### 🔍 Pipeline Telemetry & Diagnostics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("HyDE Retrieved", diagnostics.get("hyde_retrieved", 0))
                    with col2:
                        st.metric("CRAG Passed", diagnostics.get("crag_passed", 0))
                    with col3:
                        st.metric("Final Reranked", diagnostics.get("reranked_final", 0))
                    with col4:
                        st.metric("CRAG Status", diagnostics.get("crag_status", "N/A"))
                        
                    with st.expander("📄 View Raw Diagnostics JSON"):
                        st.json(diagnostics)
                        
                else:
                    st.error(f"Backend API Error [{response.status_code}]: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend: {e}")