import os
import shutil
import openai
import langchain
import pinecone 
import requests
import PyPDF2
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import chainlit as cl
from langchain import OpenAI
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain_community.vectorstores import Chroma
from openai import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
embedding_functions = SentenceTransformerEmbeddings(model_name= "all-MiniLM-L6-v2")

# Load environment variables from the .env file
load_dotenv()


# Retrieve the values
model = os.getenv('model')
api_key = os.getenv('api_key')


welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF file
2. Ask a question about the file
"""


SAVE_DIRECTORY = ".\Documnt\chroma_niraj"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)


# Global variable to store the vectorstore
vectorstore = None

# Function to parse PDF and split text into smaller chunks
def read_pdf_as_documents(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        document = Document(
            metadata={'source': pdf_path, 'page': page_num},
            page_content=text.strip()
        )
        documents.append(document)
    return documents

# Store embeddings in ChromaDB
def store_embeddings(documents):
    global vectorstore  # Use the global variable
    vectorstore = Chroma.from_documents(documents, embedding_functions, persist_directory="./chroma_db_new")

# llm client

llm = AzureChatOpenAI(model_name=model, openai_api_version="2023-07-01-preview",azure_endpoint="https://llmops-classroom-openai.openai.azure.com/",temperature=0.5,openai_api_key=api_key)


# Call LLM with the user query and relevant text
def call_LLM(user_question):
    global vectorstore  # Use the global variable
    # Retrieve relevant text directly in this function
    relevant_texts = vectorstore.similarity_search(user_question, k=2)  # Adjust k as needed
    
    if relevant_texts:
        # Extract the content from the retrieved documents
        relevant_text = "\n".join([doc.page_content for doc in relevant_texts])
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,  # Ensure llm is defined and initialized properly
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        full_prompt = f"{user_question}\n\nRelevant Information:\n{relevant_text}"
        response = chain.run(full_prompt)
    else:
        response = "Sorry, I couldn't find any relevant information."

    return response



@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message, accept=["application/pdf"]
        ).send()

    pdf_file = files[0]

    # Read the PDF file and get the documents
    documents = read_pdf_as_documents(pdf_file.path)

    # Save the uploaded PDF to the specified location
    destination_path = os.path.join(SAVE_DIRECTORY, pdf_file.name)
    shutil.copy(pdf_file.path, destination_path)

    # Store the extracted documents as embeddings in ChromaDB
    store_embeddings(documents)

    # Notify the user that the document has been processed
    await cl.Message(content="The document has been processed. You can now ask questions.").send()

@cl.on_message
async def handle_user_query(message):
    user_question = message.content  # Get the user's question
    
    # Call LLM with the user question directly
    answer = call_LLM(user_question)

    # Send the response back to the user
    await cl.Message(content=answer).send()