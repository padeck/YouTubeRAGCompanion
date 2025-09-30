# main.py

import os
from dotenv import load_dotenv

# --- Structured Output Parsers ---
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List

# --- LangChain Core Imports ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LangChain Community/Partner Imports ---
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Load environment variables (for your API keys)
load_dotenv()
# Make sure you have OPENAI_API_KEY in your .env file or environment

# ==============================================================================
# MODULE 1: DATA LOADING
# ==============================================================================

def load_transcript_from_youtube(url: str) -> list[Document]:
    """
    Loads the transcript from a YouTube URL and returns it as a LangChain Document.
    Handles potential errors if transcripts are disabled.
    
    Args:
        url: The YouTube video URL.

    Returns:
        A list containing a single Document with the transcript text, or an empty list if it fails.
    """
    if "=" in url:
        video_id = url.split("=")[-1]
    else:
        video_id = url.split("/")[-1]

    docs = []
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['de', 'en'])
        transcript_text = " ".join([item.text for item in transcript])
        print("-> Transkript erfolgreich geladen.")
        docs = [Document(page_content=transcript_text, metadata={"source": url})]
        return docs
    except TranscriptsDisabled:
        print(f"FEHLER: Für das Video {url} sind die Transkripte deaktiviert.")

# ==============================================================================
# MODULE 2: INDEXING AND RETRIEVAL SETUP
# ==============================================================================

def create_vector_retriever(docs: list[Document]):
    """
    Takes a list of documents, splits them, creates embeddings, and sets up a vector store retriever.
    
    Args:
        docs: A list of LangChain Documents.

    Returns:
        A Chroma vector store retriever object.
    """
    print("2. Splitting transcript into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"-> Transcript split into {len(splits)} chunks.")

    print("3. Creating embeddings and storing in Chroma vector database...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("-> Vector database is ready.")
    
    return vectorstore.as_retriever()

# ==============================================================================
# MODULE 3: CHAIN CREATION
# ==============================================================================

def create_rag_chain(retriever):
    """
    Creates the RAG (Retrieval-Augmented Generation) chain using LCEL.
    
    Args:
        retriever: The vector store retriever to use for context.

    Returns:
        A runnable RAG chain.
    """
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatOpenAI(model="gpt-4o") # Using a more modern model
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

# ==============================================================================
# MODULE 4: SUMMARIZATION CHAIN CREATION
# ==============================================================================


def create_summarizer_chains(model, text_splitter): # <-- Pass in the text_splitter
    """
    Creates a set of chains for generating different summary formats.
    """
    # --- The Map-Reduce Chain for long documents ---
    # This is the robust way to summarize
    map_reduce_chain = load_summarize_chain(
        llm=model,
        chain_type="map_reduce",
        verbose=True # Set to True to see the inner workings!
    )

    # --- The simpler chains for other tasks (we can keep these for now) ---
    bullet_point_prompt = ChatPromptTemplate.from_template(
        "You are an expert analyst. Extract the key points from the following video transcript and present them as a detailed, nested bulleted list:\n\n{transcript}"
    )

    # THE FIX: Restore the full prompt string here
    tweet_thread_prompt = ChatPromptTemplate.from_template(
        """You are a social media marketing expert. Your goal is to create a viral tweet thread based on the content of this video transcript.
        Rules:
        - The thread must have at least 3 tweets.
        - Each tweet must be under 280 characters.
        - Use engaging language, emojis, and relevant hashtags.
        - Start the thread with a strong hook.

        Video Transcript:
        {transcript}"""
    )

    def create_chain_from_prompt(prompt):
        return (
            {"transcript": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
    
    return {
        "map_reduce": map_reduce_chain, 
        "bullets": create_chain_from_prompt(bullet_point_prompt),
        "tweets": create_chain_from_prompt(tweet_thread_prompt),
    }
# ==============================================================================
# MAIN EXECUTION SCRIPT
# This is where you orchestrate the calls to your modules.
# ==============================================================================

'''
if __name__ == "__main__":
    # Load environment variables from .env
    load_dotenv()

    # Access your API key
    OPENAI_API_KEY = os.getenv("API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    #YOUTUBE_URL = "https://www.youtube.com/watch?v=HfrCcKDzOow" 
    #YOUTUBE_URL = "https://www.youtube.com/watch?v=3FNZdixeuZw"
    
    # 1. Load the data
    documents = load_transcript_from_youtube(YOUTUBE_URL)
    
    # Only proceed if documents were loaded successfully
    if documents:
        # Get the full transcript text for our summarizers
        full_transcript = " ".join([doc.page_content for doc in documents])

        # --- Setup for Q&A ---
        print("\n--- Setting up RAG for Q&A ---")
        retriever = create_vector_retriever(documents)
        qa_chain = create_rag_chain(retriever)

        # --- Text Splitter ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        # --- NEW: Setup for Summarization ---
        print("\n--- Setting up Summarizer Chains ---")
        # We can create one model instance and share it
        llm_model = ChatOpenAI(model="gpt-4o")
        summarizers = create_summarizer_chains(llm_model, text_splitter)

        # --- Setup for Extraction ---
        print("\n--- Setting up Extraction Chain ---")
        #extraction_chain = create_extraction_chain(llm_model)

        # 4. Use the chains!
        print("\n--- The YouTube Companion is ready! ---")
        
        # --- Example Q&A ---
        print("\n--- Testing Q&A ---")
        question = "What is ReAct?"
        print(f"\n[Question]: {question}")
        answer = qa_chain.invoke(question)
        print(f"[Answer]: {answer}")

        # --- NEW: Example Summaries ---
        print("\n--- Testing Summaries ---")
        
        print("\n[Summary Type]: Map Reduce Summary")
        map_reduce_summary_result = summarizers["map_reduce"].invoke(splits)
        print(map_reduce_summary_result['output_text'])


        print("\n[Summary Type]: Key Bullet Points")
        bullet_summary = summarizers["bullets"].invoke(full_transcript)
        print(bullet_summary)

        # --- Example Extraction ---
        #print("\n--- Testing Structured Data Extraction ---")
        #extracted_data = extraction_chain.invoke(full_transcript)
        
        #print("\n[Extracted Data]:")
        #print(extracted_data.model_dump())
'''

def create_extraction_chain(model):
    """
    Creates a chain that extracts structured entities from a transcript.
    
    Args:
        model: The chat model to use for extraction.

    Returns:
        A runnable chain that outputs a Pydantic object.
    """
    parser = PydanticOutputParser(pydantic_object=VideoEntities)

    prompt = ChatPromptTemplate.from_template(
        """
        Analyze the following video transcript and extract the key entities mentioned.
        You must follow the formatting instructions precisely.

        {format_instructions}

        Video Transcript:
        {transcript}
        """,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    extraction_chain = (
        {"transcript": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    return extraction_chain

class VideoEntities(BaseModel):
    """A structured representation of key entities mentioned in a video transcript."""
    topics: List[str] = Field(description="A list of the main technical or conceptual topics discussed in the video.")
    tools: List[str] = Field(description="A list of any software, libraries, or specific tools mentioned.")
    people: List[str] = Field(description="A list of the names of any people mentioned in the video.")
