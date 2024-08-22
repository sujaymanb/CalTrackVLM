''' old requirements
!pip install -q langchain
!pip install -q torch
!pip install -q transformers
!pip install -q sentence-transformers
!pip install -q datasets
!pip install -q faiss-cpu
'''

#from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from transformers import AutoTokenizer, AutoModelForQuestionAnswering
#from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
#import glob
import os


class RAG:
    def __init__(self,
            docpath="./data",
            embedding_model_path="sentence-transformers/all-MiniLM-l6-v2",
            qa_model_path="Intel/dynamic_tinybert",
            debug=True):
        self.debug = debug

        # init loader
        # self.loader = HuggingFaceDatasetLoader(
        #                 dataset_name, 
        #                 column_name)

        documents = []

        for file in os.listdir(docpath):
            if file.endswith('.txt'):
                path = os.path.join(docpath, file)
                documents.extend(TextLoader(path).load())

        # split into docs
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        documents = text_splitter.split_documents(documents)

        # load embeddings models
        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        self.embeddings = HuggingFaceEmbeddings(
                            model_name=embedding_model_path,
                            model_kwargs=model_kwargs,
                            encode_kwargs=encode_kwargs)

        # vector database
        print("loading vector database...")
        self.db = Chroma.from_documents(documents, self.embeddings)

        # # load the QA model
        # tokenizer = AutoTokenizer.from_pretrained(qa_model_path, padding=True, truncation=True, max_length=512)
        # self.llm = pipeline(
        #     "question-answering", 
        #     model=qa_model_path, 
        #     tokenizer=tokenizer,
        #     return_tensors='pt'
        # )

        # retrieval chain
        self.retriever = self.db.as_retriever(search_kwargs={"k": 4})
        #self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="refine", retriever=self.retriever, return_source_documents=False)

        # debug logs
        # if self.debug:
        #     print(data[:2])
        #     print(docs[0])


    def get_embeddings(self, text):
        return self.embeddings.embed_query(text)


    def db_search(self, prompt, return_all = False):
        result_docs = self.db.similarity_search(prompt)

        if return_all:
            return result_docs

        return result_docs[0].page_content


    def retrieve(self, prompt, k=1, search_type="mmr"):
        retriever = self.db.as_retriever(search_type=search_type)
        search_kwargs={"k": k}
        docs = retriever.invoke(prompt)

        return docs

    # def run(self, prompt, return_all = False):
    #     result = self.qa_chain.run({"query": prompt})
    #     if return_all:
    #         return result

    #     return result["result"]


def main():
    # init rag
    rag = RAG()

    # test embeddings
    text = "Grilled Cheese Nutrition"
    result = rag.get_embeddings(text)

    print(f"embeddings:\n{result[:3]}")

    # test vector db search
    result = rag.db_search(text)

    print(f"search results:\n{result}")

    # # invoke chain
    # result = rag.run("Who is George Washington?")
    # print(result)

main()
