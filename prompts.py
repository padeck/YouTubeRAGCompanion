RAG_PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

BULLET_SUMMARY_PROMPT_TEMPLATE = """
You are an expert analyst. Extract the key points from the following
video transcript and present them as a detailed, nested bulleted list.
Focus on the main arguments, important data points, and conclusions.

Video Transcript:
{transcript}
"""

TWEET_THREAD_PROMPT_TEMPLATE = """
You are a social media marketing expert. Your goal is to create a
viral tweet thread based on the content of this video transcript.
Rules:
- The thread must have at least 3 tweets.
- Each tweet must be under 280 characters.
- Use engaging language, emojis, and relevant hashtags.
- Start the thread with a strong hook to grab attention.

Video Transcript:
{transcript}
"""

ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
Analyze the following video transcript and extract the key entities mentioned.
You must follow the formatting instructions precisely.

{format_instructions}

Video Transcript:
{transcript}
"""
