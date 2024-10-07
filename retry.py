import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:  # Only add non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_llama.pkl"

main_placeholder = st.empty()

# Load Hugging Face API token from environment
access_token = "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"

# Initialize the LLaMA 2 pipeline with the token
pipe = pipeline("text-generation", model="meta-llama/Llama-2-13b-f", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-f", token=access_token)
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-f", token=access_token)

# Asynchronous URL loader
async def fetch_url_content(session, url):
    async with session.get(url) as response:
        return await response.text()

async def load_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_content(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Parallel processing of documents
def process_document(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    return text_splitter.split_documents([doc])

if process_url_clicked:
    if not urls:
        st.error("Please enter at least one URL before processing.")
    else:
        try:
            # Asynchronous data loading
            main_placeholder.text("Data Loading...Started...✅✅✅")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            contents = loop.run_until_complete(load_urls(urls))
            data = [Document(page_content=content, metadata={"source": url}) for content, url in zip(contents, urls)]

            if not data:
                st.error("No data was loaded from the provided URLs. Please check the URLs and try again.")
            else:
                # Parallel document splitting
                main_placeholder.text("Text Splitting...Started...✅✅✅")
                with ThreadPoolExecutor() as executor:
                    split_docs = list(executor.map(process_document, data))
                docs = [doc for sublist in split_docs for doc in sublist]

                if not docs:
                    st.error("No documents were created after splitting. The content might be too short or empty.")
                else:
                    # Create embeddings using HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    
                    # Debug: Print the shape of the first embedding
                    first_embedding = embeddings.embed_documents([docs[0].page_content])
                    st.write(f"Shape of first embedding: {len(first_embedding[0])}")
                    
                    vectorstore_llama = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

                    # Save the FAISS index to a pickle file
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_llama, f)
                    
                    st.success("URLs processed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Use st.cache_data to cache the question-answering function
@st.cache_data
def answer_question(query, file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    return chain({"question": query}, return_only_outputs=True)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            result = answer_question(query, file_path)
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
    else:
        st.error("Please process some URLs before asking questions.")
