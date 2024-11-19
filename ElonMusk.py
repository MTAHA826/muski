
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['qdrant_api_key'] = os.getenv('qdrant_api_key')
os.environ['google_api_key'] = os.getenv('google_api_key')

# Streamlit UI setup
st.title("Musk Chatbot")
query = st.text_input("Please enter your query:")

# Load documents
loader = WebBaseLoader(
    'https://en.wikipedia.org/wiki/Elon_Musk',
    bs_kwargs=dict(parse_only=SoupStrainer(class_='mw-content-ltr mw-parser-output'))
)
documents = loader.load()

# Split documents into chunks
recursive = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = recursive.split_documents(documents)

# Initialize embeddings and vector store
embed = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embed,
    url='https://1328bf7c-9693-4c14-a04c-f342030f3b52.us-east4-0.gcp.cloud.qdrant.io:6333',
    api_key=os.getenv('qdrant_api_key'),
    prefer_grpc=True,
    collection_name="Elon Muske"
)

# Initialize LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv('google_api_key')
)

# Prompt template
prompt_str = """
You are a highly knowledgeable and conversational chatbot
specializing in providing accurate and insightful 
information about Elon Musk. Answer all questions with detail
as if you are an expert on his life, career, companies, and achievements.
context: {context}

Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)
num_chunks = 5
retriever = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})

# Function to stream responses
def generate_response(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    formatted_prompt = _prompt.format(question=query, context=context)
    
    # Use the LLM's streaming method
    for chunk in llm.stream(formatted_prompt):
        yield chunk  # Yield each chunk of the response

# Chatbot interaction
if query:
    st.subheader("Chatbot Response:")
    placeholder = st.empty()  # Create a placeholder for the streaming response
    response = ""

    # Stream response chunk by chunk
    for chunk in generate_response(query):
        response += chunk
        placeholder.markdown(response)  # Update the placeholder with the new chunk
else:
    st.write("Please enter a query to interact with the chatbot.")
