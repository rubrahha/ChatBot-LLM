import streamlit as st
import os
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline


with st.sidebar:
    st.title('LLM CHAT APP')
    st.markdown(''' 
        ## About
        This app is an LLM-powered chatbot built using:
        - [Python]
        - [Streamlit]
        - [Langchain]
        - [OpenAI / Hugging Face]
    ''')
    add_vertical_space(5)
    st.write('Made with love by Shivam 💖')

def main():
    st.header("📄 Chat with PDF")

    load_dotenv()  # Load env vars

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        store_name = pdf.name[:-4]

        VECTOR_DIR = "vectorstores"
        os.makedirs(VECTOR_DIR, exist_ok=True)
        file_path = os.path.join(VECTOR_DIR, f"{store_name}.pkl")

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                VectorStore = pickle.load(f)
            st.success("✅ Loaded existing vector store.")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(file_path, "wb") as f:
                pickle.dump(VectorStore, f)
            st.success("✅ Created and saved new vector store.")

        query = st.text_input("Ask your questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Use a smaller and faster model
            pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=128,
                temperature=0.5,
            )
            llm = HuggingFacePipeline(pipeline=pipe)

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == '__main__':
    main()