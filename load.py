from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer





loader = TextLoader("bank.pdf", encoding="utf8")
documents = loader.load()

print(f"Loaded {len(documents)} documents" )

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print(f"Split into {len(texts)} chunks of text")

for i, text in enumerate(texts):
    print(f"Chunk {i+1}: {text.page_content[:100]}...")  # Print the first 100 characters of each chunk


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in texts]
print(f"Generating embeddings for {len(texts)} chunks of text")
embeddings = embedding_model.encode(texts)  
print(f"Generated embeddings for {len(embeddings)} chunks of text")

print(f"Embedding for first chunk: {embeddings}...")  # Print the first 10 dimensions of the embedding for the first chunk

