from dotenv import load_dotenv

# --- Structured Output Parsers ---
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List, Dict, Literal

# --- LangChain Core Imports ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LangChain Community/Partner Imports ---
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter
)
from langchain.chains.summarize import load_summarize_chain

from prompts import (
    RAG_PROMPT_TEMPLATE,
    BULLET_SUMMARY_PROMPT_TEMPLATE,
    TWEET_THREAD_PROMPT_TEMPLATE,
    ENTITY_EXTRACTION_PROMPT_TEMPLATE
)

# Load environment variables (for your API keys)
load_dotenv()
# Make sure you have OPENAI_API_KEY in your .env file or environment


# ==============================================================================
# 1. DATA STRUCTURES (Structured Output Schemas)
# ==============================================================================

class VideoEntities(BaseModel):
    """
    Structured representation of key entities mentioned in a video transcript.
    """
    topics: List[str] = Field(
        description="A list of the main technical or conceptual "
                    "topics discussed in the video."
    )
    tools: List[str] = Field(
        description="A list of any software, libraries, "
                    "or specific tools mentioned."
    )
    people: List[str] = Field(
        description="A list of the names of any people mentioned in the video."
    )


# ==============================================================================
# 2. CORE LOGIC (The Processor Class)
# ==============================================================================

class YouTubeProcessor:
    """
    A class to process YouTube videos, enabling RAG-based querying,
    summarization, and entity extraction.
    """
    def __init__(
        self,
        chat_model: ChatOpenAI = None,
        embeddings_model: OpenAIEmbeddings = None,
        text_splitter: TextSplitter = None,
    ):
        """
        Initializes the processor with configurable LangChain components.
        """
        self.model = chat_model or ChatOpenAI(model="gpt-4o")
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

        # State variables that will be populated after loading a video
        self.docs: List[Document] = []
        self.retriever = None
        self.full_transcript_text: str = ""

    def load_video(self, url: str) -> bool:
        """
        Loads and processes a YouTube video transcript. This includes:
        1. Fetching the transcript.
        2. Splitting it into chunks.
        3. Creating a vector store retriever.

        Args:
            url: The YouTube video URL.

        Returns:
            True if loading was successful, False otherwise.
        """
        print(f"\n--- Loading and Processing Video: {url} ---")
        if not self._load_transcript_from_youtube(url):
            return False

        self._create_vector_retriever()
        return True

    def _load_transcript_from_youtube(self, url: str) -> bool:
        """
        Loads the transcript from a YouTube URL and returns it as
        a LangChain Document.
        Handles potential errors if transcripts are disabled.

        Args:
            url: The YouTube video URL.

        Returns:
            A list containing a single Document with the transcript text,
            or an empty list if it fails.
        """
        if "=" in url:
            video_id = url.split("=")[-1]
        else:
            video_id = url.split("/")[-1]

        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=['de', 'en'])
            transcript_text = " ".join([item.text for item in transcript])
            self.full_transcript_text = transcript_text
            self.docs = [
                Document(page_content=self.full_transcript_text,
                         metadata={"source": url})
                ]
            print("-> Transcript loaded successfully.")
            return True
        except TranscriptsDisabled:
            print("Error: Transcripts seem to be disabled"
                  " for the provided video.")
            return False

    def _create_vector_retriever(self):
        """
        Takes a list of documents, splits them, creates embeddings,
        and sets up a vector store retriever.
        """
        print("2. Splitting transcript into chunks...")
        # --- USE THE INSTANCE'S SPLITTER ---
        splits = self.text_splitter.split_documents(self.docs)
        print(f"-> Transcript split into {len(splits)} chunks.")

        print("3. Creating embeddings and storing in Chroma vector database..")
        # --- USE THE INSTANCE'S EMBEDDINGS ---
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.embeddings
        )
        print("-> Vector database is ready.")

        self.retriever = vectorstore.as_retriever()

    def query(self, question: str) -> str:
        """
        Performs Retrieval-Augmented Generation (RAG) to answer a question
        based on the loaded video transcript.
        """
        if not self.retriever:
            return (
                "Error: Please load a video first using "
                "the 'load_video' method."
            )

        print(f"\n🔍 Querying: '{question}'")

        rag_chain = self._create_rag_chain()
        result = rag_chain.invoke(question)

        print("✅ RAG query complete.")
        return result

    def summarize(
        self,
        summary_type: Literal["map_reduce", "bullets", "tweets"],
    ) -> str:
        """
        Generates a summary of the video transcript in the specified format.
        """
        if not self.docs:
            return (
                "Error: Please load a video first "
                "using the 'load_video' method."
            )

        print(f"\n📄 Generating summary (type: {summary_type})...")

        summarizer_chains = self._create_summarizer_chains()
        if summary_type not in summarizer_chains:
            return (
                f"Error: Invalid summary type '{summary_type}'."
                f" Available types: {list(summarizer_chains.keys())}"
            )

        chain = summarizer_chains[summary_type]

        # The map_reduce chain expects a list of documents.
        if summary_type == "map_reduce":
            input_data = self.docs
            result = chain.invoke({"input_documents": input_data})
        else:
            input_data = self.full_transcript_text
            result = chain.invoke({input_data})

        # The map_reduce chain returns a dict, others return a string
        if isinstance(result, dict):
            output = result.get('output_text', result)
        else:
            output = result

        print("✅ Summary generated.")
        return output

    def extract_entities(self) -> VideoEntities | str:
        """
        Extracts key entities (topics, tools, people) from the transcript.
        """
        if not self.full_transcript_text:
            return (
                "Error: Please load a video first"
                "using the 'load_video' method."
            )

        print("\n🔎 Extracting entities...")
        extraction_chain = self._create_extraction_chain()
        result = extraction_chain.invoke(self.full_transcript_text)
        print("✅ Entities extracted.")
        return result

    # -- Chain Creation Methods ---

    def _create_rag_chain(self) -> Runnable:
        """Creates the RAG chain."""
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )

    def _create_summarizer_chains(self) -> Dict[str, Runnable]:
        """Creates and returns a dictionary of summarization chains."""
        # Map-Reduce for long documents
        map_reduce_chain = load_summarize_chain(
            llm=self.model, chain_type="map_reduce"
        )

        # Simple chain for bullet points
        bullet_point_prompt = ChatPromptTemplate.from_template(
            BULLET_SUMMARY_PROMPT_TEMPLATE
        )
        bullet_chain = bullet_point_prompt | self.model | StrOutputParser()

        # Simple chain for tweet threads
        tweet_thread_prompt = ChatPromptTemplate.from_template(
            TWEET_THREAD_PROMPT_TEMPLATE
        )
        tweet_chain = tweet_thread_prompt | self.model | StrOutputParser()

        return {
            "map_reduce": map_reduce_chain,
            "bullets": RunnablePassthrough() | {
                "transcript": RunnablePassthrough()
            } | bullet_chain,
            "tweets": RunnablePassthrough() | {
                "transcript": RunnablePassthrough()
            } | tweet_chain,
        }

    def _create_extraction_chain(self) -> Runnable:
        """Creates the structured entity extraction chain."""
        parser = PydanticOutputParser(pydantic_object=VideoEntities)
        prompt = ChatPromptTemplate.from_template(
            ENTITY_EXTRACTION_PROMPT_TEMPLATE,
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )
        return (
            {"transcript": RunnablePassthrough()}
            | prompt
            | self.model
            | parser
        )

# ==============================================================================
# 3. MAIN EXECUTION SCRIPT
# ==============================================================================


def main():
    """
    Main function to demonstrate the YouTubeProcessor's capabilities.
    """
    # URL of a video to analyze (e.g., a short LangChain tutorial)
    YOUTUBE_URL = "https://www.youtube.com/watch?v=sY6pI1S0de8"

    # 1. Initialize the processor
    processor = YouTubeProcessor()

    # 2. Load the video (this also creates the retriever)
    if not processor.load_video(YOUTUBE_URL):
        print("Exiting due to failure in video processing.")
        return

    # 3. Perform RAG-based Q&A
    question = "What is LCEL?"
    answer = processor.query(question)
    print("\n--- RAG Answer ---")
    print(f"Q: {question}\nA: {answer}")
    print("-" * 20)

    # 4. Generate Summaries
    bullet_summary = processor.summarize("bullets")
    print("\n--- Bullet Point Summary ---")
    print(bullet_summary)
    print("-" * 20)

    # 5. Extract Structured Entities
    entities = processor.extract_entities()
    print("\n--- Extracted Entities ---")
    if isinstance(entities, VideoEntities):
        print(f"Topics: {entities.topics}")
        print(f"Tools: {entities.tools}")
        print(f"People: {entities.people}")
    else:
        print(entities)  # Print error message if it failed
    print("-" * 20)


if __name__ == "__main__":
    main()
