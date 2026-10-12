from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
import os
token = os.getenv("GROQ_API_KEY")


data = TextLoader('./RAG_project/text.txt')
docs = data.load()


template = ChatPromptTemplate.from_messages([
    ("system","you are ai for summurizes the text "),
    ("human","{data}")
])


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=token
)

promt = template.format_messages(data = docs[0].page_content)



  
response = llm.invoke(promt)

print(response.content)