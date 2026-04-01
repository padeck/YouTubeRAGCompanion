import streamlit as st
import os
from backend import YouTubeProcessor, VideoEntities

# --- Helper Functions for UI Rendering ---

def render_qa_tab(processor: YouTubeProcessor):
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
    st.header("Generate Summaries")
    summary_options = {
        "Overall Summary": "map_reduce",
        "Key Bullet Points": "bullets",
        "Tweet Thread": "tweets",
    }
    summary_choice = st.selectbox("Choose summary type:",
                                  summary_options.keys(),
                                  key="summary_choice")

    if st.button("Generate Summary", key="summary_button"):
        summary_type_key = summary_options[summary_choice]
        with st.spinner(f"Generating {summary_choice}..."):
            summary = processor.summarize(summary_type_key)
            st.markdown(summary)

def render_extraction_tab(processor: YouTubeProcessor):
    st.header("Extract Structured Data")
    if st.button("Extract Entities", key="extract_button"):
        with st.spinner("Extracting entities..."):
            extracted_data = processor.extract_entities()
            if isinstance(extracted_data, VideoEntities):
                st.json(extracted_data.model_dump())
            else:
                st.error(extracted_data)

# --- Main App Logic ---

def main():
    st.set_page_config(page_title="Ultimate YouTube Companion", layout="wide")
    st.title("🤖 Ultimate YouTube Video Companion")
    st.markdown(
        "Provide a YouTube URL and get answers, "
        "summaries, and structured data completely locally."
    )

    if "processor" not in st.session_state:
        st.session_state.processor = None

    youtube_url = st.text_input(
        "Enter YouTube URL:", placeholder="https://www.youtube.com/..."
    )

    if st.button("Process Video", key="process_button", disabled=not youtube_url):
        if youtube_url:
            with st.spinner("Fetching transcript and building local vector index..."):
                if st.session_state.processor and st.session_state.processor.vectorstore:
                    try:
                        st.session_state.processor.vectorstore.delete_collection()
                    except Exception:
                        pass
                processor = YouTubeProcessor()
                success = processor.load_video(youtube_url)
                if success:
                    st.session_state.processor = processor
                    st.success("Video processed successfully!")
                else:
                    st.session_state.processor = None
                    st.error("Failed to load or process the video transcript.")
        else:
            st.warning("Please enter a YouTube URL.")

    if st.session_state.processor:
        view_options = ["❓ Q&A", "📄 Summarizer", "📊 Data Extractor"]
        current_view = st.radio(
            "Choose a feature",
            options=view_options,
            key="navigation",
            horizontal=True,
            label_visibility="collapsed"
        )

        if current_view == "❓ Q&A":
            render_qa_tab(st.session_state.processor)
        elif current_view == "📄 Summarizer":
            render_summary_tab(st.session_state.processor)
        elif current_view == "📊 Data Extractor":
            render_extraction_tab(st.session_state.processor)


if __name__ == "__main__":
    main()