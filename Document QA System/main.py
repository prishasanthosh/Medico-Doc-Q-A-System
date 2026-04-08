import os
import fitz
from dotenv import load_dotenv
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


pdf_dir="data"
chroma_db_path="chroma_db"
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")


def load_and_extract_pdf(pdf_path):
    document_data=[]
    for file in os.listdir(pdf_path):
        if file.endswith(".pdf"):
            file_path=os.path.join(pdf_path,file)
            document=fitz.open(file_path)
            data=""
            for side in document:
                data+=side.get_text()
            document_data.append((file,data))
    return document_data

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunk=[]
    for file, data in docs:
        split=text_splitter.split_text(data)
        for i, chunks in enumerate(split):
            chunk.append(Document(page_content=chunks,metadata={"source": file, "chunk_id": i}))
    return chunk
        
def create_vector_store(docs,db_path):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
    )
    vectordb=Chroma.from_documents(docs, embedding=embeddings,persist_directory=db_path)
    return vectordb

def create_qa_chain(vectordb):
    retrieve=vectordb.as_retriever(search_type="similarity",search_kwargs={"k":4})
    llm=ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro-preview-03-25",
        temperature=0.2,
    )
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retrieve,
        return_source_documents=True,
    )
    return chain

print("Extract the PDF Contents")
documents = load_and_extract_pdf(pdf_dir)

print("Split the text into chunks")
chunks = chunk_documents(documents)

print("Create the vector database for the chunks")
vectordb = create_vector_store(chunks, chroma_db_path)

print("Initializing the Medico Doc Q&A system")
qa_chain = create_qa_chain(vectordb)

def question_box(query):
    if not query:
        return "Please enter a question.", ""
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = "\n".join(
        f"→ {doc.metadata['source']} (chunk {doc.metadata['chunk_id']})"
        for doc in result["source_documents"]
    )
    return answer, sources

interface = gr.Interface(
    fn=question_box,
    inputs=gr.Textbox(lines=2, placeholder="Ask the query about your Medical PDFs..."),
    outputs=["text", "text"],
    title="📚 Medico Doc Q&A with Gemini",
    description="Ask questions from your medical documents ang get answers using Google Gemini",
    theme="soft"
)

if __name__ == "__main__":
    interface.launch()

