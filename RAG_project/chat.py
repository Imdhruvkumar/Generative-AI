from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
import os
token = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=token
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma-DB",
    embedding_function=embedding_model
)

retriver = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lemda_mult":0.5
    }
)

## prompt 

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's question only using the provided document context. If the answer is not available in the context, clearly say 'I could not find the answer in the provided document.' Do not make up or assume any information."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

print("Rag system created")

print("press 0 to exit ")

while True:
    query = input("you : ")
    if query == 0:
        break
    docs = retriver.invoke(query)


    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })

    response = llm.invoke(final_prompt)

    print("\n AI: ",response.content)