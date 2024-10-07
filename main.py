import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env



# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 30px;
        color: #FF5733;
        text-align: center;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stTextInput {
        border-radius: 10px;
        border: 2px solid #FF5733;
        padding: 10px;
    }
    
    .st-emotion-cache-lpgk4i:hover {
        background-color: white;
        font-color: white;
    }
    .st-emotion-cache-lpgk4i {
        background-color: #FF5733; /* Change button background color */
        color: white; /* Change text color */
        border-radius: 10px; /* Add border radius */
        padding: 10px 0; /* Add padding */
        width: 100%; /* Make button full width */
        border: none; /* Remove border */
        transition: background-color 0.3s ease; /* Smooth transition */
    }
    
    </s
    </style>
    """,
    unsafe_allow_html=True
)



# Configure Google GenerativeAI
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.title("GenAI News Research Tool")
st.write("---")  
st.sidebar.title("News Article URLs")


url1 = st.sidebar.text_input("URL 1")

url2 = st.sidebar.text_input("URL 2")
    
url3 = st.sidebar.text_input("URL 3")
urls = [url for url in [url1, url2, url3] if url]  # Collect non-empty URLs


process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)

if process_url_clicked:
    if not urls:
        st.error("Please enter at least one URL before processing.")
    else:
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started....")
            data = loader.load()

            # Debug: Print loaded data
            for i, doc in enumerate(data):
                print(f"Document {i + 1} from URL {urls[i]} loaded successfully!")
                print(f"Content: {doc.page_content[:500]}")  # Print the first 500 characters for preview

            if not data:
                st.error("No data was loaded from the provided URLs. Please check the URLs and try again.")
            else:
                # Split data
                main_placeholder.text("Text Splitter...Started........")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)

                if not docs:
                    st.error("No documents were created after splitting. The content might be too short or empty.")
                else:
                    # Create embeddings using HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    # Debug: Print the shape of the first embedding
                    first_embedding = embeddings.embed_documents([docs[0].page_content])
                    st.write(f"Shape of first embedding: {len(first_embedding[0])}")

                    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Embedding Vector Started Building.....")
                    time.sleep(2)

                    # Save the FAISS index to a pickle file
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_gemini, f)

                    st.success("URLs processed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(f"Error occurred while loading or processing URLs: {str(e)}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
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
            print(f"Error during question-answering: {str(e)}")
    else:
        st.error("Please process some URLs before asking questions.")
