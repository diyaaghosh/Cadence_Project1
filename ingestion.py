from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil



def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path) # valid for only digital pdfs
    return loader.load()

def split_documents_into_chunk(documents,chunk_size=1000,chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". "," ",""])
    return splitter.split_documents( documents)

def process_pdfs(files,retriever):
    if os.path.exists("temp"): # store pdfs in temp folder
        shutil.rmtree("temp")
    os.makedirs("temp",exist_ok=True)
    all_documents = []
    for file in files:
        path = os.path.join("temp",file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        documents = load_pdf(path)
        print(file.name,"pages:",len(documents))
        for doc in documents:
            doc.metadata["source_file"] = file.name
            doc.metadata["page"] = (doc.metadata.get("page",0) + 1)
        all_documents.extend( documents)
    print("Extracted pages:",len(all_documents))
    all_documents = [doc for doc in all_documents if doc.page_content.strip()]
    print("Pages with text:",len(all_documents))
    chunks = split_documents_into_chunk(all_documents)
    print("Chunks:",len(chunks))
    if not chunks:
        raise ValueError( "No text found. PDF may need OCR.")
    retriever.add_documents(chunks)
    print("Added",len(chunks),"chunks")
    return chunks