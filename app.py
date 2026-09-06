import streamlit as st
from rag_engine import Advanced_Rag
from retriever import RAGRetriever
from VectorStore import vector_store
from embedding import Embedding_Manager
from llm import get_llm
from ingestion import process_pdfs



st.set_page_config(page_title="PDF AI Assistant",layout="wide")

@st.cache_resource
def load_pipeline():
    embedding_manager = Embedding_Manager()
    store = vector_store()
    retriever = RAGRetriever( store, embedding_manager)
    llm = get_llm()
    rag = Advanced_Rag(rag_retriever=retriever,llm=llm)
    return rag

rag = load_pipeline()
# clear the pdfs (old) so that it don't mixed up
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    rag.clear_history()
    try:
        rag.rag_retriever.vector_store.clear()
        print("Old vectors cleared")

    except Exception as e:
        print("Clear skipped:", e)



st.sidebar.title(" Upload PDFs")
pdfs = st.sidebar.file_uploader("Upload PDF files",type=["pdf"],accept_multiple_files=True)

if pdfs:
    if st.sidebar.button("Process Documents"):
        with st.spinner("Creating embeddings..." ):
            process_pdfs(pdfs,rag.rag_retriever)
        st.sidebar.success(f"{len(pdfs)} PDFs processed")

st.title(" PDF AI Assistant")

tabs = st.tabs([" Chat"])

with tabs[0]:
    question = st.chat_input("Ask from your PDFs...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message( "assistant"):
            result = rag.query(question,top_k=5,stream=False,)
            st.write(result["Answer"] )
            with st.expander(" Sources"):
                for src in result["Sources"]:
                    st.write(
# print the citation , source and pages 
                    f""" 
                     {src['source']}
                    Page:
                    {src['page']}
                    Similarity:
                    {src['score']:.3f}
                    {src['preview']}
                    """
                    )

# summary
