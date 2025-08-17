# Loads the data log in the JSON file.
with open("input.txt", "r") as file:
    data = json.load(file)

# Flattens the logs into DataFrame, parses timestamps, and filters for occupied logs.   
df = pd.DataFrame(data)
df = df.drop(columns=["id", "timestamp"])
df = df[df["status"] == "occupied"]

# Converts each log row into a readable summary strings.
def summarize_log(row):
    return f"Log {row['log_id']} is occupied by {row['user']} from {row['start_time']} to {row['end_time']}."

# Turns each summary into a LangChain Document for embeddings.
documents = [Document(page_content=summarize_log(row)) for _, row in df.iterrows()]

# Embeds the documents using Ollama, stores them in ChromaDB, and persist the database.
embedding = ollama.Embedding(model="your-model-name")
chroma_dir = "path/to/chroma_db"

vector_store = ChromaDB(embedding_function=embedding, persist_directory=chroma_dir)
vector_store.add_documents(documents)
vector_store.persist()

# Sets up the Ollama LLM and retrieval
llm = ollama.LLM(model="your-model-name")
qa_chain = RetrievalQA(llm=llm, retriever=vector_store.as_retriever(),chain_type="stuff")