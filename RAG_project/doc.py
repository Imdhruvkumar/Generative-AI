from langchain_community.document_loaders import TextLoader

data = TextLoader(
    './RAG_project/text.txt'
)

doc = data.load()

print(doc[0])