from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

short_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in 1 or 2 lines"
)
long_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in simple words"
)

model = ChatGroq( model="openai/gpt-oss-20b")

parser = StrOutputParser()
### sequence  runnable####
# chains = prompt | model | parser

## parlel runnable

chains =RunnableParallel( {
    "short":short_prompt | model | parser,
    "long":long_prompt| model | parser
})
result = chains.invoke({"topic":"machine learning"})
# formetted_prompt = prompt.format_messages(topic= "machine learning")

# response = model.invoke(formetted_prompt)

# result = parser.parse(response.content)
print(result)