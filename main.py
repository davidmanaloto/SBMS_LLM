import json
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

with open("room_logs.json", "r") as f: data = json.load(f)

df = pd.json_normalize(data["logs"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df[df["occupancy_status"]== "occupied"]

def row_to_summary(row):
    return (
        f"At {row['timestamp']}, the room was occupied with "
        f"{row['occupancy_count']} people. Energy: {row['energy_consumption_kwh']} kWh. "
        f"Lighting: {row['power_consumption_watts.lighting']}W, "
        f"HVAC: {row['power_consumption_watts.hvac_fan']}W, "
        f"Standby: {row['power_consumption_watts.standby_misc']}W, "
        f"Total Power: {row['power_consumption_watts.total']}W. "
        f"Lights on: {row['equipment_usage.lights_on_hours']}h, "
        f"AC on: {row['equipment_usage.air_conditioner_on_hours']}h, "
        f"Projector: {row['equipment_usage.projector_on_hours']}h, "
        f"Computers: {row['equipment_usage.computer_on_hours']}h. "
        f"Temp: {row['environmental_data.temperature_celsius']}°C, "
        f"Humidity: {row['environmental_data.humidity_percent']}%."
    )
documents = [Document(page_content=row_to_summary(row), metadata={"timestamp": row["timestamp"].isoformat()}) for _, row in df.iterrows()]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
chroma_dir = "./chroma_room_logs"

vectorstore = Chroma.from_documents(documents=documents, embeddings=embeddings, persist_directory=chroma_dir,collection_name="room_logs")
vectorstore.persist()

llm = Ollama(model="llama3.1:8b")

RetrievalQAChain = RetrievalQA.from_chain_type(llm=llm,retriever=vectorstore.as_retriever(), chain_type="stuff", return_source_documents=True)

while True:
    query = input("\n Ask a question about the room logs (or type 'exit'): ")
    if query.lower() in {"exit", "quit"}:
        break
    answer = RetrievalQAChain.run({query})
    print("\nAnswer:", answer)