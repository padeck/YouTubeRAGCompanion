import streamlit as st
import os
# Import all the functions from your backend file
from backend import (
    load_transcript_from_youtube,
    create_vector_retriever,
    create_rag_chain,
    create_summarizer_chains,
    create_extraction_chain
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OPENAI_API_KEY not found in Streamlit secrets. Please add it to your .streamlit/secrets.toml file.")
    st.stop() # Stop the app if the key is not found

# ==============================================================================
#      STREAMLIT CACHING: This is the most important part!
# ==============================================================================
# This decorator tells Streamlit to run this function only once and cache the result.
# If the function is called again with the same input (URL), it will return the
# cached data instead of re-processing everything.
@st.cache_resource
def load_and_prepare_video_data(_url):
    """
    A cached function to handle all the expensive, one-time setup steps.
    Takes a URL, returns all necessary components for the app to function.
    """
    # 1. Load data
    documents = load_transcript_from_youtube(_url)
    if not documents:
        return None # Return None if loading fails

    # 2. Prepare for RAG (Splitting & Vector Store)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    retriever = create_vector_retriever(splits) # Assuming create_vector_retriever handles this
    
    # 3. Prepare full transcript for other tasks
    full_transcript = " ".join([doc.page_content for doc in documents])

    # 4. Create all chains
    llm_model = ChatOpenAI(model="gpt-4o")
    qa_chain = create_rag_chain(retriever)
    summarizers = create_summarizer_chains(llm_model, text_splitter)
    extraction_chain = create_extraction_chain(llm_model)

    return {
        "full_transcript": full_transcript,
        "splits": splits,
        "qa_chain": qa_chain,
        "summarizers": summarizers,
        "extraction_chain": extraction_chain,
    }

# ==============================================================================
#                                THE APP'S UI
# ==============================================================================

st.set_page_config(page_title="Ultimate YouTube Companion", layout="wide")
st.title("🤖 Ultimate YouTube Video Companion")
st.markdown("Provide a YouTube URL and get answers, summaries, and structured data.")

# --- 1. User Input ---
youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=3FNZdixeuZw")

if youtube_url:
    with st.spinner("Fetching transcript and preparing chains... This may take a moment."):
        video_data = load_and_prepare_video_data(youtube_url)

    if video_data is None:
        st.error("Failed to load transcript. Please check the URL and ensure the video has transcripts.")
    else:
        st.success("Video processed successfully! You can now use the tools below.")
        
        # --- 2. Create Tabs for Different Tools ---
        q_and_a_tab, summary_tab, extraction_tab = st.tabs(["❓ Q&A", "📄 Summarizer", "📊 Data Extractor"])

        with q_and_a_tab:
            st.header("Ask Questions About the Video")
            question = st.text_input("Your Question:", key="qa_question")
            if st.button("Get Answer", key="qa_button"):
                if question:
                    with st.spinner("Finding answer..."):
                        answer = video_data["qa_chain"].invoke(question)
                        st.write(answer)
                else:
                    st.warning("Please enter a question.")

        with summary_tab:
            st.header("Generate Summaries")
            summary_type = st.selectbox(
                "Choose summary type:", [
                    "Overall Summary (Map-Reduce)",
                    "Key Bullet Points",
                    "Tweet Thread"])
            
            if st.button("Generate Summary", key="summary_button"):
                with st.spinner(f"Generating {summary_type}..."):
                    if summary_type == "Overall Summary (Map-Reduce)":
                        summary = video_data["summarizers"]["map_reduce"].invoke(
                            video_data["splits"]
                            )
                        st.markdown(summary['output_text'])
                    elif summary_type == "Key Bullet Points":
                        summary = video_data["summarizers"]["bullets"].invoke(
                            video_data["full_transcript"]
                            )
                        st.markdown(summary)
                    elif summary_type == "Tweet Thread":
                        summary = video_data["summarizers"]["tweets"].invoke(
                            video_data["full_transcript"]
                            )
                        st.markdown(summary)

        with extraction_tab:
            st.header("Extract Structured Data")
            if st.button("Extract Entities", key="extract_button"):
                with st.spinner("Extracting entities..."):
                    extracted_data = video_data["extraction_chain"].invoke(
                        video_data["full_transcript"]
                    )
                    st.json(extracted_data.model_dump())