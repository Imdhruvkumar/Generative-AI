from dotenv import load_dotenv
import os

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# llm = HuggingFaceEndpoint(
#     repo_id="CohereLabs/tiny-aya-earth:cohere",
#     temperature=0.7,
#     huggingfacehub_api_token=token,
# )

# model = ChatHuggingFace(llm=llm)


from langchain.chat_models import init_chat_model

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model = init_chat_model(
    "CohereLabs/tiny-aya-earth:cohere",
    model_provider="huggingface",
    temperature=0.7,
    
    huggingfacehub_api_token=token,
)


response = model.invoke("define ml in two lines")

print(response.content)