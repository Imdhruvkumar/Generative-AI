from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain.tools import tool
from rich import print



# creating a tool
@tool
def get_text_length(text:str) -> int:
    """ return the number of character in given text"""

    print(len(text))


model = ChatGroq( model="openai/gpt-oss-20b")

# tool binding to the model
model_withtool = model.bind_tools([get_text_length])
# result = model.invoke("hello")
result = model_withtool.invoke("return the number of character in given text : how are you")



print(result.tool_calls[0])
print(get_text_length.invoke({
    'name': 'get_text_length',
    'args': {'text': 'how are you'},
    'id': 'fc_06c49773-1a7d-4c7a-ac2f-f67a79c8dc45',
    'type': 'tool_call'
}))