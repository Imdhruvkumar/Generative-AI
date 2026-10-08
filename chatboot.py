from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging

logging.set_verbosity_error()

pipe = pipeline(
    "text-generation",
    model="./models/Qwen2.5-1.5B-Instruct"
)

list = []

while True:
    print("----------for exit chat write 0")
    
    promt = input("you : ")
    if promt == "0":
        break

    list.append(promt)
    
    messages = [
    {
    "role":"user",
    "content":promt
    }   
    ]

    result = pipe(
    messages,
  
    )
    

    print("Bot : ",result[0]["generated_text"][-1]["content"])

    #done