# 첫 시작은 랭체인 docs에서 Modules>Retrieval>Document>loaders>PDF...
# 고민하지말고 구글링, 언어모델 docs의 quick start를 보자!
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


#시각화위해 streamlit 활용
import streamlit as st
import tempfile
import os

#제목
st.title("chatPDF")
st.write("---")

#파일업로드
uploaded_file = st.file_uploader("PDF파일을 업로드에 해주세요", type=["pdf"])
st.write("---")

#구글링으로 찾아낸, pdf upload시 임시로 저장하는 함수
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드되면 동작하는코드 
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100, #절대적 글자수
        chunk_overlap  = 20, #겹침 허용구간..안어색하게 자르게
        length_function = len,
        is_separator_regex = False, #정규표현식이라면 T
    ) #이까지가 vector화, 이걸 embedding해서 숫자화 해야함



    texts = text_splitter.split_documents(pages)

    #embedding, .env에서 키따로 사용
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma.. 무료 embedding 모델
    db = Chroma.from_documents(texts, embeddings_model)

    #question
    st.header("ChatPDF에게 무엇이든 질문해 주세요")
    question = st.text_input("업로드한 PDF와 관계된 질문을 입력하세요")
    
       
    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            prompt_template = """You are an expert in document summaries, and you can answer any questions about the posts in the pdf file. If the question is not in the pdf file, answer that you don't know, and don't make it up and answer it.
            {context}
            Question: {question}
            """
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"])
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            chain_type_kwargs = {"prompt": PROMPT}
            qa_chain = RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)
            result = qa_chain({"query": question})
            st.write(result["result"])

