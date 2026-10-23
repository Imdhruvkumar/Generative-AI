from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

prompt = ChatPromptTemplate.from_template(
    """
    you are a helpful assistent

    summarise the news in crear bullet points

    {news}
"""
)

model = ChatGroq( model="openai/gpt-oss-20b")

parser = StrOutputParser()

search_tool = TavilySearch(max_result=5) 

chain = prompt | model | parser 

response = search_tool.invoke({
    "query": "Latest  AI developments 2026"
})

news_result = chain.invoke({"news":response})

print(news_result)