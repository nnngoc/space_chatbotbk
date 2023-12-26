from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
import os
import re
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from langchain.schema.retriever import BaseRetriever, Document
from typing import List
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores import VectorStore
from llm import URALLM
from langchain.prompts import PromptTemplate

# Get role for  passage document
def get_role(document):
    """
    Get role for student.
    """
    # Tìm kiếm các từ khóa liên quan đến vai trò học viên trong document.
    keywords = [
        "sinh viên",
        "đại học",
        "học viên",
        "thạc sĩ",
        "nghiên cứu sinh",
        "tiến sĩ",
    ]
    role = []
    for keyword in keywords:
        if keyword in document.metadata['source'].lower():
            role.append(keyword)
    return ", ".join(role)

def processing_data(data_path):
    folders = os.listdir(data_path)

    dir_loaders = []

    # Add the documents to the project
    for folder in folders:
        dir_loader = DirectoryLoader((os.path.join(data_path, folder)), loader_cls=TextLoader)
        dir_loaders.append(dir_loader)

    # Load the text files.
    loaded_documents = []
    for dir_loader in dir_loaders:
        loaded_documents.append(dir_loader.load())

    data = []
    for i in range(len(loaded_documents)):
        for j in range(len(loaded_documents[i])):
            data.append(loaded_documents[i][j])

    # Final data prepare for vector database
    for document in data:
        role = get_role(document)
        document.metadata['role'] = role
    
    return data

# Embedding model
# embedding = HuggingFaceEmbeddings(
#     model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
#     model_kwargs={"device": "cpu"}
# )

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
)

# Vector database
data_path = 'raw_data'
persist_directory = 'vector_db'
vectordb = Chroma.from_documents(
    documents=processing_data(data_path),
    embedding=embedding,
    persist_directory=persist_directory
)

class CustomRetriever(BaseRetriever):
    vectorstores:Chroma
    retriever:vectordb.as_retriever()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Use your existing retriever to get the documents
        documents = self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())

        # Get page content
        docs_content = []
        for i in range(len(documents)):
          docs_content.append(documents[i].page_content)

        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # So we create the respective sentence combinations
        sentence_combinations = [[query, document] for document in docs_content]

        # Compute the similarity scores for these combinations
        similarity_scores = model.predict(sentence_combinations)

        # Sort the scores in decreasing order
        sim_scores_argsort = reversed(np.argsort(similarity_scores))

        # Store the rerank document in new list
        docs = []
        for idx in sim_scores_argsort:
          docs.append(documents[idx])

        docs_top_4 = docs[0:4]

        return docs_top_4
    
llm = URALLM()
custom_retriever = CustomRetriever(vectorstores = vectordb,retriever = vectordb.as_retriever(search_kwargs={"k": 50}))

# Build prompt
template = """[INST] <<SYS>>

Bạn là một chatbot hỗ trợ các quy định học vụ của trường Đại học Bách Khoa - ĐHQG TP.HCM.
Bạn sử dụng văn bản được cung cấp để trả lời câu hỏi cho người dùng.
Không sử dụng bất kỳ thông tin nào khác ngoài văn bản đã cho.
Trả lời đầy đủ và ngắn gọn nhất có thể.
Không đề cập tên riêng trong câu trả lời.
Không chứa các ký tự: "-/", "]", "/", "-" trong câu trả lời.

<</SYS>>

Văn bản: {context}
Câu hỏi: {question}
Câu trả lời:
[/INST]"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,
                                       verbose=False,
                                       # retriever=vectordb.as_retriever(),
                                       retriever=custom_retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

def remove_special_characters(text):
    text = text.replace('].', '')
    text = text.replace('/.', '')
    text = text.replace('/.-', '')
    return text

def rag(question: str) -> str:
    # call QA chain
    response = qa_chain({"query": question})

    return remove_special_characters(response["result"])

