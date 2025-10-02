# YouTube Video Companion

This is a powerful Streamlit application that acts as an intelligent companion for YouTube videos that come with subtitles. By simply providing a YouTube URL, you can unlock a suite of AI-powered tools to ask questions, generate various types of summaries, and extract structured data from the video's content.

The application leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers based on the video's transcript.

## Features

* **❓ Interactive Q&A:** Ask specific questions about the video content and get answers generated directly from the transcript.
* **📄 Advanced Summarization:** Choose from multiple summary formats:
  * **Overall Summary:** A concise, high-level overview of the video.
  * **Key Bullet Points:** A structured, nested list of the most important points and arguments.
  * **Tweet Thread:** An engaging, ready-to-post Twitter thread complete with emojis and hashtags.
* **📊 Structured Data Extraction:** Automatically extract key entities from the video—such as topics, tools, and people mentioned—and view them in a clean JSON format.
* **⚙️ Efficient Backend:** Built with LangChain, the backend fetches transcripts, chunks them, creates vector embeddings, and orchestrates calls to the OpenAI API.

## Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Core AI/LLM Framework:** [LangChain](https://www.langchain.com/)
* **LLM & Embeddings:** [OpenAI (GPT-4o)](https://openai.com/)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (in-memory)
* **Transcript Fetching:** [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/)
* **Data Validation:** [Pydantic](https://pydantic.dev/) (for structured output)

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

* Python 3.9+
* An [OpenAI API Key](https://platform.openai.com/api-keys)
* An [LangChain API Key](https://www.langchain.com/langchain)

### 2. Clone the Repository

```bash
git clone https://github.com/padeck/YouTubeRAGCompanion.git
cd RAG
```

### 3. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

Install dependencies using pip:
`pip install -r requirements.txt`

### 5. Configure Your API Keys

You need to provide your OpenAI API key, as well as your LangChain API key. The application is configured to look for it in an environment variable.

Create a file named `secrets.toml` in a directory called `.streamlit` which should be placed in the root of the project.

In the `secrets.toml`-file, provide both API keys as such:

```bash
OPENAI_API_KEY = "sk-..."
LANGCHAIN_API_KEY = "..."
```

### 6. How to Run

With your virtual environment activated and the `secrets.toml` file created, you can now run the Streamlit application:

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually <http://localhost:8501>).

### 7. Project Structure

```
├── .streamlit/
│ └──secrets.toml       # For storing local environment variables (API keys).
├── app.py              # The main Streamlit frontend application.
├── backend.py          # Core logic with the YouTubeProcessor class.
├── prompts.py          # Contains all LLM prompt templates.
├── README.md           # This markdown file
└── requirements.txt    # List of Python dependencies.
```

* `app.py`: Handles all the user interface components using Streamlit. It takes user input (YouTube URL, questions), calls the backend processor, and displays the results

* `backend.py`: Contains the YouTubeProcessor class, which encapsulates all the heavy lifting: fetching transcripts, creating the RAG pipeline, and defining the summarization and extraction chains

* `prompts.py`: Separates the prompts from the application logic, making them easy to view and modify

### 8. How it works (High Level)

The application follows a standard RAG (Retrieval-Augmented Generation) pattern:

1. Ingestion & Processing: When a user provides a YouTube URL, the YouTubeProcessor:

    * Fetches the video's full transcript using youtube-transcript-api.

    * Uses a LangChain TextSplitter to break the transcript into smaller, manageable chunks.

    * Generates vector embeddings for each chunk using OpenAI's embeddings model.

    * Stores these chunks and their embeddings in an in-memory Chroma vector database.

2. Application Layer:

    * For Q&A: When a user asks a question, the application performs a similarity search in the Chroma database to find the most relevant chunks of the transcript. These chunks (the context) are then passed along with the user's question to the LLM to generate an answer.

    * For Summarization & Extraction: The full transcript is passed to different, purpose-built LangChain chains that use specific prompts to either summarize the content or extract entities into a structured Pydantic model.

### 9. How it works (Deep Dive)

The entire process can be broken down into two main phases: the **Indexing Phase** (processing the video) and the **Application Phase** (Q&A, Summarizing, etc.).

#### Phase 1: The Indexing Phase (When you click "Process Video")

This phase is all about converting the unstructured audio/video content into a structured, searchable knowledge base. This is handled by the `YouTubeProcessor.load_video()` method.

**Step 1: Transcript Fetching**

* **What it does**: The `_load_transcript_from_youtube method` is called. It extracts the unique video_id from the URL.
* **How it works**: It uses the `youtube-transcript-api` library to connect to YouTube's backend and request the closed captions/transcript for that video. It prioritizes English (`en`) but will fall back to German (`de`) if available.
* **The Output**: A single, long string of text containing the entire video transcript (e.g., `self.full_transcript_text`). This raw text is also wrapped in a LangChain Document object, which is a standard way to handle text data in the framework.

**Step 2: Text Splitting (Chunking)**

* **The Problem**: LLMs a limited "context window" (a maximum number of tokens they can read at once). A 30-minute video transcript is far too long to fit into a single prompt. Furthermore, for Q&A, you only need the small, specific parts of the transcript relevant to the user's question, not the whole thing.

* **What it does**: The `_create_vector_retriever` method uses a `RecursiveCharacterTextSplitter`.

* **How it works**: This splitter intelligently breaks the long transcript into smaller, overlapping chunks.

  * `chunk_size=1000`: It tries to make each chunk about 1000 characters long.

  * `chunk_overlap=100`: Each new chunk starts 100 characters before the previous one ended. This is crucial to prevent losing context. For example, if a sentence is split exactly between two chunks, the overlap ensures the full sentence is captured in at least one of the chunks.

**Step 3: Vectorization (Creating Embeddings)**

* **The Goal**: We need a way for the computer to understand the semantic meaning of each text chunk, not just the words themselves. This is where embeddings come in.

* **What it doe**s: For every single text chunk created in Step 2, the system calls the OpenAI Embeddings API via `OpenAIEmbeddings()`.

* **How it works**: The embeddings model converts each text chunk into a high-dimensional vector. The key property of these vectors is that chunks with similar meanings will have vectors that are mathematically close to each other (usually cosine similarity).

**Step 4: Storing in a Vector Database**

* **What it does**: The system uses `Chroma.from_documents()`.

* **How it works**: Chroma is a vector database. It takes all the text chunks and their corresponding vector embeddings and stores them together. It creates a special index that makes it incredibly fast to search for vectors.

* **The Result**: At the end of this phase, `st.session_state.processor` holds an object that now contains a fully indexed, searchable version of the video's content, ready to be used. The `self.retriever` object is now a gateway to this database.

#### Phase 2: The Application Phase (Interacting with the UI)

1. **Query Vectorization**: The user's question is also converted into a vector using the same OpenAI embeddings model.
2. **Similarity Search**: The `retriever` takes the question's vector and searches the Chroma database to find the text chunks whose vectors are closest to it. This is the "Retrieval" part of RAG.
3. **Context Stuffing**: The retriever returns the top N most relevant chunks of the original transcript.
4. **Augmented Prompting**: These retrieved chunks (the "context") are inserted into the `RAG_PROMPT_TEMPLATE` along with the original question.
5. **Generation**: The final, augmented prompt is sent to the LLM (`self.model`). Because the model has been given the exact text from the video, it can generate a factual, grounded answer. This prevents hallucination and ensures the answer comes directly from the source material.

#### The Map-Reduce Pattern

The "Overall Summary" option uses a `map_reduce` chain. This is a classic and powerful pattern for processing documents that are too large for a single LLM call.

1. **The "Map" Step**

    * **Goal**: Create an initial summary for every small piece of the document independently and in parallel.

    * **How it works**:

        1. The `load_summarize_chain` takes the list of all the document chunks we created during the indexing phase.

        2. It iterates through each chunk one by one.

        3. For **each chunk**, it sends a request to the LLM with a prompt like: `"Summarize the following text: [text of chunk 1]"`.

        4. Repeat the same process for every chunk.

    * **The Output of this step is a list of many small summaries**. We now have a "summary of the summary" problem, but the total length of these combined summaries is much shorter than the original transcript.
2. **The "Reduce" (or "Combine") Step**
    * **Goal**: Take all the intermediate summaries and combine them into one final, coherent summary.

    * **How it works**:

        1. The chain now takes all the summaries generated in the "Map" step.

        2. It joins them together into a single block of text.

        3. It makes a **final call** to the LLM with a different prompt, something like: "The following is a set of summaries from a document. Synthesize them into a final, consolidated summary: [summary 1 text] [summary 2 text] [summary 3 text] ..."

    * **The final output is the single, high-level summary of the entire video transcript.**
