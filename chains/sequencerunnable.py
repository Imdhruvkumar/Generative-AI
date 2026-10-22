from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "explain {topic} in simple words"
)

model = ChatGroq( model="openai/gpt-oss-20b")

parser = StrOutputParser()

formetted_prompt = prompt.format_messages(topic= "machine learning")

response = model.invoke(formetted_prompt)

result = parser.parse(response.content)
print(result)