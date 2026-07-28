import streamlit as st
import requests
import asyncio
import aiohttp
import json

st.set_page_config(page_title="Self-Healing RAG System", page_icon="🛡️", layout="wide")
st.title("🛡️ Advanced Self-Healing RAG Engine")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Backend Config")
    # Replace this with your actual Render URL
    # api_url = st.text_input("FastAPI Endpoint", value="https://self-healing-rag-chatbot.onrender.com")
    api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000")

# --- Define UI Tabs ---
tab1, tab2 = st.tabs(["⚖️ A/B Comparison Mode (Live Streaming)", "🛡️ Standard Mode"])

# ==========================================
# TAB 1: A/B COMPARISON MODE 
# ==========================================
with tab1:
    st.caption("Executes both pipelines asynchronously, showing live telemetry step-by-step.")
    query_compare = st.text_input("Ask a research question to compare:", key="q_comp")
    
    # Reads the streaming JSON lines and appends them to a visible log
    async def fetch_and_render_stream(session, url, payload, ui_container, title, color):
        with ui_container.container():
            st.subheader(f"{color} {title}")
            
            # Create a collapsible expander for the "Thinking" steps
            with st.expander("💭 Thinking Process", expanded=True):
                log_container = st.empty()
                log_text = ""
            
            # Placeholder for the final answer below the thinking logs
            answer_container = st.empty()
            
            async with session.post(url, json=payload) as response:
                async for line in response.content:
                    if not line:
                        continue
                        
                    data = json.loads(line.decode('utf-8'))
                    
                    # If the chunk contains a "step", append it to our running log
                    if "step" in data:
                        log_text += f"✔️ {data['step']}\n\n"
                        log_container.markdown(log_text)
                        
                    # If the chunk contains the final "answer", render it below
                    elif "answer" in data:
                        with answer_container.container():
                            st.success(data["answer"])
                            st.caption(f"⏱️ Latency: {data.get('latency', 'N/A')}s")
                        return data

    async def run_ab_test(query):
        col1, col2 = st.columns(2)
        container_naive = col1.container()
        container_sh = col2.container()
        placeholder_judge = st.empty()
        
        async with aiohttp.ClientSession() as session:
            payload = {"query": query}
            
            # Fire both streams concurrently
            task_naive = fetch_and_render_stream(session, f"{api_url}/api/chat/naive", payload, container_naive, "Naive RAG", "⚡")
            task_sh = fetch_and_render_stream(session, f"{api_url}/api/chat/self-healing", payload, container_sh, "Self-Healing RAG", "🛡️")
            
            res_naive, res_sh = await asyncio.gather(task_naive, task_sh)
            
            # Trigger Judge
            placeholder_judge.warning("⚖️ Both pipelines complete. Groq Judge is evaluating results...")
            judge_payload = {"query": query, "normal_res": res_naive, "sh_res": res_sh}
            
            async with session.post(f"{api_url}/api/judge", json=judge_payload) as judge_res:
                judge_data = await judge_res.json()
                with placeholder_judge.container():
                    st.divider()
                    st.header("⚖️ LLM Judge Verdict")
                    st.info(judge_data["verdict"])

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