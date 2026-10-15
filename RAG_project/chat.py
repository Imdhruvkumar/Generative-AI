from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
import os
token = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=token
)


# response = llm.invoke(promt)

# print(response.content)