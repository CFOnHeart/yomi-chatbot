from src.embeddings.azure_openai_embeddings import get_azure_openai_embeddings

embeddings = get_azure_openai_embeddings()

print (embeddings.embed_documents("Hi, I am Bob"))