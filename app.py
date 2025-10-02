import streamlit as st
import os
from backend import YouTubeProcessor, VideoEntities

# --- Helper Functions for UI Rendering ---


def render_qa_tab(processor: YouTubeProcessor):
    """Renders the Q&A tab UI and handles its logic."""
    st.header("Ask Questions About the Video")
    question = st.text_input("Your Question:", key="qa_question")
    if st.button("Get Answer", key="qa_button", disabled=not question):
        if question:
            with st.spinner("Finding answer..."):
                answer = processor.query(question)
                st.write(answer)
        else:
            st.warning("Please enter a question.")


def render_summary_tab(processor: YouTubeProcessor):
    """Renders the Summarizer tab UI and handles its logic."""
    st.header("Generate Summaries")
    summary_options = {
        "Overall Summary": "map_reduce",
        "Key Bullet Points": "bullets",
        "Tweet Thread": "tweets",
    }
    summary_choice = st.selectbox("Choose summary type:",
                                  summary_options.keys())

    if st.button("Generate Summary", key="summary_button"):
        summary_type_key = summary_options[summary_choice]
        with st.spinner(f"Generating {summary_choice}..."):
            summary = processor.summarize(summary_type_key)
            st.markdown(summary)


def render_extraction_tab(processor: YouTubeProcessor):
    """Renders the Data Extractor tab UI and handles its logic."""
    st.header("Extract Structured Data")
    if st.button("Extract Entities", key="extract_button"):
        with st.spinner("Extracting entities..."):
            # Call the processor's extraction method
            extracted_data = processor.extract_entities()
            if isinstance(extracted_data, VideoEntities):
                st.json(extracted_data.model_dump())
            else:
                st.error(extracted_data)

# --- Main App Logic ---


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Ultimate YouTube Companion", layout="wide")
    st.title("🤖 Ultimate YouTube Video Companion")
    st.markdown(
        "Provide a YouTube URL and get answers, "
        "summaries, and structured data."
    )

    # Use session_state to store the processor object across reruns
    if "processor" not in st.session_state:
        st.session_state.processor = None

    youtube_url = st.text_input(
        "Enter YouTube URL:", placeholder="https://www.youtube.com/..."
    )

    if st.button("Process Video", key="process_button", disabled=not youtube_url):
        if youtube_url:
            with st.spinner(
                "Fetching transcript and building vector index..."
            ):
                # 1. Create an instance of the processor
                processor = YouTubeProcessor()
                # 2. Load the video (this handles all setup)
                success = processor.load_video(youtube_url)
                # 3. Store the entire processor object in the session state
                if success:
                    st.session_state.processor = processor
                    st.success("Video processed successfully!")
                else:
                    st.session_state.processor = None
                    st.error("Failed to load or process the video transcript.")
        else:
            st.warning("Please enter a YouTube URL.")

    # Render UI tabs only if the processor object exists in the session state
    if st.session_state.processor:
        q_and_a_tab, summary_tab, extraction_tab = st.tabs(
            ["❓ Q&A", "📄 Summarizer", "📊 Data Extractor"]
        )
        with q_and_a_tab:
            render_qa_tab(st.session_state.processor)
        with summary_tab:
            render_summary_tab(st.session_state.processor)
        with extraction_tab:
            render_extraction_tab(st.session_state.processor)


if __name__ == "__main__":
    # --- Load API Key from Streamlit Secrets ---
    # This part remains the same. The processor will use the env var.
    try:
        # Check for secrets first, for deployment
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (AttributeError, KeyError):
        # Fallback for local development if secrets aren't set
        from dotenv import load_dotenv
        load_dotenv()
        if "OPENAI_API_KEY" not in os.environ:
            st.error("OPENAI_API_KEY not found. Please set it in your "
                     "environment variables or Streamlit secrets.")
            st.stop()

    main()
