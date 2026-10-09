from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging
from langchain_core.output_parsers import PydanticOutputParser
logging.set_verbosity_error()

class Movie(BaseModel):
    title: str
    summary: str 
    key_points: list[str] 
    lesson: str 
    release_year:Optional[int]
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]


parser = PydanticOutputParser(pydantic_object=Movie)

pipe = pipeline(
    "text-generation",
    model="./models/Qwen2.5-1.5B-Instruct"
)

promt = PromptTemplate(
    template="""Extract movie information from the paragraph.

{format_instructions}

Paragraph: {paragraph}

Return the structured movie information:""",
    input_variables=["paragraph"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

paragraph = input("Give me the paragraph for summury :-")


final_promt = promt.format(
    paragraph=paragraph
    )

result = pipe(final_promt)

response = result[0]["generated_text"].strip()

try:
    movie_data = parser.parse(response)

    print("\nAI - Structured Output:")
    print(movie_data.model_dump_json(indent=4))

except Exception as e:
    print("Could not parse the model output:", e)
    print("Raw response:", response)