from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

data = TextLoader(
    './RAG_project/text.txt'
)

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 100,
    chunk_overlap=1
)

chunks = splitter.split_documents(docs)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-DB"

)



























































# docs = [
#     Document(
#         page_content="Python is a programming language used for web development, automation, and artificial intelligence.",
#         metadata={"page": 1, "topic": "Python"}
#     ),
#     Document(
#         page_content="React is used to build user interfaces. Node.js and Express help developers create backend APIs for web applications.",
#         metadata={"page": 2, "topic": "Web Development"}
#     ),
#     Document(
#         page_content="RAG retrieves relevant information from documents. Embeddings convert text into numerical vectors for semantic search.",
#         metadata={"page": 3, "topic": "Generative AI"}
#     )
# ]

# embedded_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# vector_store = Chroma.from_documents(
#     documents= docs,
#     embedding=embedded_model,
#     persist_directory="chroma-db"
# )

# result = vector_store.similarity_search('what  is used for build user interfaces',k=2)

# for r in result:
#     print(r)



