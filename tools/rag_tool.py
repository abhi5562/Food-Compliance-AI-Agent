from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
import os
from uuid import uuid4

load_dotenv()

vector_store = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX"],
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
    embedding=OpenAIEmbeddings(),
    namespace="",
    text_key="text",
)

@tool(response_format="content_and_artifact")
def rag_tool(query: str):
    """Search demonstration restaurant SOPs for food safety and kitchen procedures.

    Args:
        query: A focused question including relevant food, activity, or equipment.

    Returns:
        Up to three passages with reference_id, chunk_content, and source metadata.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    context = [{
        "reference_id": uuid4().hex,
        "chunk_content": retrieved_chunk.page_content, 
        "metadata": retrieved_chunk.metadata
        }
        for retrieved_chunk in retriever.invoke(query)
    ]

    return context, context
