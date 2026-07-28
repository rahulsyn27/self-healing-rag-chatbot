import streamlit as st
import requests
import asyncio
import aiohttp
import json

st.set_page_config(page_title="Self-Healing RAG System", page_icon="🛡️", layout="wide")
st.title("🛡️ Advanced Self-Healing RAG Engine")

# ==========================================
# SIDEBAR & UPLOAD LOGIC
# ==========================================
with st.sidebar:
    # st.header("⚙️ Backend Config")
    # api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000") # to run this in local
    api_url = st.text_input("FastAPI Endpoint", value="https://self-healing-rag-chatbot.onrender.com")
    
    st.divider()
    
    st.header("📄 Upload Document")
    st.caption("Test the pipeline with your own data.")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"]) 
    
    if uploaded_file is not None: 
        st.success(f"Ready to process: {uploaded_file.name}")
        
        if st.button("Ingest into Database", type="primary", use_container_width=True):
            with st.spinner("Slicing and embedding document... This might take a minute."):
                try:
                    # Package the file for the POST request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    # Send it to the FastAPI backend
                    response = requests.post(f"{api_url}/api/ingest", files=files)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        if "error" in res_data:
                            st.error(f"Error: {res_data['error']}")
                        else:
                            st.success(res_data["message"])
                    else:
                        st.error(f"Failed to connect. Status code: {response.status_code}")
                except Exception as e:
                    st.error(f"An frontend error occurred: {str(e)}")

# ==========================================
# DYNAMIC TOP BANNER
# ==========================================
# Only show the default banner if the user hasn't uploaded a custom PDF
if uploaded_file is None:
    st.info("ℹ️ **Demo Mode Active:** The vector database is currently pre-loaded with **`fedavg.pdf`** and **`fedprox.pdf`**.\n Upload other documents in the sidebar to test")
    
    with st.expander("💡 Click here for suggested test questions", expanded=False):
        st.markdown("""
        Copy and paste any of these questions to test the RAG architectures:
        
        **1. Vocabulary Mismatch (Tests HyDE):**
        > *"Why is it better to aggregate weights on a central server rather than just uploading all the user's mobile data?"*
        
        **2. Deep Comparison (Tests Cross-Encoder):**
        > *"What is the core mathematical difference between Federated SGD and Federated Averaging?"*
        
        **3. Out-of-Scope (Tests CRAG Filtering):**
        > *"What is my current GPA?"*
        """)

# --- Define UI Tabs ---
tab1, tab2 = st.tabs(["⚖️ A/B Comparison Mode (Live Streaming)", "🛡️ Standard Mode"])

# ==========================================
# TAB 1: A/B COMPARISON MODE 
# ==========================================
with tab1:
    st.caption("Executes both pipelines asynchronously, showing live telemetry step-by-step.")
    query_compare = st.text_input("Ask a research question to compare:", key="q_comp")
    
    async def fetch_and_render_stream(session, url, payload, ui_container, title, color):
        with ui_container.container():
            st.subheader(f"{color} {title}")
            
            with st.expander("💭 View Thinking Process", expanded=True):
                log_container = st.empty()
                log_text = ""
            
            answer_container = st.empty()
            
            try:
                async with session.post(url, json=payload) as response:
                    async for line in response.content:
                        if not line:
                            continue
                            
                        try:
                            data = json.loads(line.decode('utf-8'))
                        except json.JSONDecodeError:
                            continue
                        
                        if "step" in data:
                            log_text += f"✔️ {data['step']}\n\n"
                            log_container.markdown(log_text)
                            
                        elif "answer" in data:
                            with answer_container.container():
                                if "Error" in data["answer"]:
                                    st.error(data["answer"])
                                else:
                                    st.success(data["answer"])
                                st.caption(f"⏱️ Latency: {data.get('latency', 'N/A')}s")
                            return data
            except aiohttp.client_exceptions.ClientPayloadError:
                with answer_container.container():
                    st.error("Connection lost: The FastAPI backend crashed mid-stream.")
                return {"answer": "Error", "latency": 0}
            except Exception as e:
                with answer_container.container():
                    st.error(f"Stream error: {str(e)}")
                return {"answer": "Error", "latency": 0}

    async def run_ab_test(query):
        col1, col2 = st.columns(2)
        container_naive = col1.container()
        container_sh = col2.container()
        placeholder_judge = st.empty()
        
        async with aiohttp.ClientSession() as session:
            payload = {"query": query}
            
            task_naive = fetch_and_render_stream(session, f"{api_url}/api/chat/naive", payload, container_naive, "Naive RAG", "⚡")
            task_sh = fetch_and_render_stream(session, f"{api_url}/api/chat/self-healing", payload, container_sh, "Self-Healing RAG", "🛡️")
            
            res_naive, res_sh = await asyncio.gather(task_naive, task_sh)
            
            placeholder_judge.warning("⚖️ Both pipelines complete. Groq Judge is evaluating results...")
            judge_payload = {"query": query, "normal_res": res_naive, "sh_res": res_sh}
            
            try:
                async with session.post(f"{api_url}/api/judge", json=judge_payload) as judge_res:
                    judge_data = await judge_res.json()
                    with placeholder_judge.container():
                        st.divider()
                        st.header("⚖️ LLM Judge Verdict")
                        st.info(judge_data["verdict"])
            except Exception:
                placeholder_judge.error("Judge evaluation failed to connect.")

    if st.button("Run Dynamic Comparison", key="btn_comp", type="primary"):
        if query_compare:
            asyncio.run(run_ab_test(query_compare))

# ==========================================
# TAB 2: STANDARD MODE 
# ==========================================
with tab2:
    st.caption("Standard Self-Healing Pipeline Execution.")
    query_standard = st.text_input("Ask a research question:", key="q_std")
    
    async def run_single(query):
        container = st.container()
        async with aiohttp.ClientSession() as session:
            await fetch_and_render_stream(session, f"{api_url}/api/chat/self-healing", {"query": query}, container, "Self-Healing RAG", "🛡️")

    if st.button("Run Engine", key="btn_std", type="primary"):
        if query_standard:
            asyncio.run(run_single(query_standard))