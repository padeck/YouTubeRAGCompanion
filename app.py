# app.py

import streamlit as st
import os
from backend import (
    load_transcript_from_youtube,
    create_vector_retriever,
    create_rag_chain,
    create_summarizer_chains,
    create_extraction_chain,
)
from langchain_openai import ChatOpenAI

# --- Helper Functions for UI Rendering ---

def render_qa_tab(video_data):
    """Renders the Q&A tab UI and handles its logic."""
    st.header("Ask Questions About the Video")
    question = st.text_input("Your Question:", key="qa_question")
    
    if st.button("Get Answer", key="qa_button"):
        if question:
            with st.spinner("Finding answer..."):
                answer = video_data["qa_chain"].invoke(question)
                st.write(answer)
        else:
            st.warning("Please enter a question.")

def render_summary_tab(video_data):
    """Renders the Summarizer tab UI and handles its logic."""
    st.header("Generate Summaries")
    summary_options = {
        "Overall Summary (Map-Reduce)": ("map_reduce", video_data["splits"]),
        "Key Bullet Points": ("bullets", video_data["full_transcript"]),
        "Tweet Thread": ("tweets", video_data["full_transcript"]),
    }
    summary_type = st.selectbox("Choose summary type:", summary_options.keys())
    
    if st.button("Generate Summary", key="summary_button"):
        with st.spinner(f"Generating {summary_type}..."):
            chain_key, input_data = summary_options[summary_type]
            summary_chain = video_data["summarizers"][chain_key]
            summary = summary_chain.invoke(input_data)
            
            # Handle different output formats from chains
            if isinstance(summary, dict):
                st.markdown(summary.get('output_text', 'No output text found.'))
            else:
                st.markdown(summary)

def render_extraction_tab(video_data):
    """Renders the Data Extractor tab UI and handles its logic."""
    st.header("Extract Structured Data")
    if st.button("Extract Entities", key="extract_button"):
        with st.spinner("Extracting entities..."):
            extracted_data = video_data["extraction_chain"].invoke(
                video_data["full_transcript"]
            )
            st.json(extracted_data.model_dump())

# --- Main App Logic ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Ultimate YouTube Companion", layout="wide")
    st.title("🤖 Ultimate YouTube Video Companion")
    st.markdown("Provide a YouTube URL and get answers, summaries, and structured data.")

    # REFACTOR: Use session_state to store data across reruns
    if "video_data" not in st.session_state:
        st.session_state.video_data = None

    youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/...")

    if st.button("Process Video", key="process_button"):
        if youtube_url:
            with st.spinner("Fetching transcript and preparing chains..."):
                # Load and prepare data, then store it in session_state
                documents = load_transcript_from_youtube(youtube_url)
                if documents:
                    retriever = create_vector_retriever(documents)
                    full_transcript = " ".join([doc.page_content for doc in documents])
                    llm_model = ChatOpenAI(model="gpt-4o")
                    
                    st.session_state.video_data = {
                        "full_transcript": full_transcript,
                        "splits": retriever.vectorstore.get(include=['documents'])['documents'], # A way to get splits back
                        "qa_chain": create_rag_chain(retriever),
                        "summarizers": create_summarizer_chains(llm_model, None), # TextSplitter not needed here anymore
                        "extraction_chain": create_extraction_chain(llm_model),
                    }
                    st.success("Video processed successfully!")
                else:
                    st.session_state.video_data = None
                    st.error("Failed to load transcript.")
        else:
            st.warning("Please enter a YouTube URL.")

    # Render UI tabs only if data is successfully loaded and stored
    if st.session_state.video_data:
        q_and_a_tab, summary_tab, extraction_tab = st.tabs(["❓ Q&A", "📄 Summarizer", "📊 Data Extractor"])
        with q_and_a_tab:
            render_qa_tab(st.session_state.video_data)
        with summary_tab:
            render_summary_tab(st.session_state.video_data)
        with extraction_tab:
            render_extraction_tab(st.session_state.video_data)

if __name__ == "__main__":
    # --- Load API Key from Streamlit Secrets ---
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OPENAI_API_KEY not found in secrets. Please add it to your .streamlit/secrets.toml")
        st.stop()
    
    main()