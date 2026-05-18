from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd

#  1 Title,Date,Rating,Review 
#  "name","reviewCount","rating","published","title","content"  
df = pd.read_csv("/Users/thor/Git/DALF26W20/home.csv", encoding="latin1", delimiter=",")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_home_langchain5_db"
add_documents = not os.path.exists(db_location)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        text = str(row["content"]) + " " + str(row["title"])

        chunks = text_splitter.split_text(text)
  
        for j, chunk in enumerate(chunks):
            document = Document(
                page_content=chunk,
                metadata={
                    "rating": row["rating"],
                    "date": row["published"],
                    "username": row["name"],
                    "reviewcount": row["reviewCount"],
                    "source_row": i
                },
                id=f"{i}_{j}"
            )

            documents.append(document)
            ids.append(f"{i}_{j}")

        
vector_store = Chroma(
    collection_name="home_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# make vectorstore usable by our llm  
retriever = vector_store.as_retriever( search_kwargs={"k": 10})
