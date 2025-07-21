from src.model.embedding.azure_openai_embeddings import YomiAzureOpenAIEmbedding

embeddings = YomiAzureOpenAIEmbedding()

print (embeddings.embed_documents(["Hi, I am Bob"]))