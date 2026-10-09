from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate

from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging

logging.set_verbosity_error()

pipe = pipeline(
    "text-generation",
    model="./models/Qwen2.5-1.5B-Instruct"
)

promt = PromptTemplate.from_template(
        """You are a helpful AI assistant.
Summarize the following paragraph in simple and clear English.
Keep the summary short and include only the main ideas.

Paragraph: {paragraph}

Summary:"""
)

paragraph = input("Give me the paragraph for summury :-")


final_promt = promt.format(paragraph=paragraph)

result = pipe(final_promt)

print("\nAI:", result[0]["generated_text"][len(final_promt):].strip())