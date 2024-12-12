import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# from langchain_openai import OpenAIEmbeddings

# from langchain_openai import OpenAIEmbeddings
api_key = ''


st.title('Document Summarizer')

uploaded_file = st.file_uploader('Upload your CV', type='pdf')

text = ""

# research to see if the is a better way to load the pdf document directly using Langchain document loaders
#  pypdf loader takes a file_path as a string. Which means you cannot directly pass the pdf file itself
if uploaded_file is not None:
    
    pdfFile = PdfReader(uploaded_file)
    
    for page in pdfFile.pages:
        text += page.extract_text()
        
    # st.write(text)
    f = open('cv.txt', 'w')
    f.write(text)
    f.close()
       

    loader = TextLoader('./cv.txt')

    # This is a long document we can split up.
    with open("./cv.txt") as f:
        CV = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.create_documents([CV])

    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    
    vector = FAISS.from_documents(documents, embeddings)

    retriever = vector.as_retriever()
    
    model = ChatMistralAI(mistral_api_key=api_key)
   
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(model, prompt)
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": "Can you help me optimize the following cv"})
    
    st.write(response["answer"])
