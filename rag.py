# Required imports
import json
import time
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from tqdm.auto import tqdm
import streamlit as st

# Constants (hardcoded)
FILE_PATH = "anjibot_chunks.json"
BATCH_SIZE = 384
INDEX_NAME = "groq-llama-3-rag"
PINECONE_API_KEY = st.secrets["keys"]["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["keys"]["GROQ_API_KEY"]
DIMS = 768
encoder = SentenceTransformer('dwzhu/e5-base-4k')
groq_client = Groq(api_key=GROQ_API_KEY)

with open(FILE_PATH, 'r') as file:
        data= json.load(file)

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region='us-east-1')
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# Check if index already exists; if not, create it
if INDEX_NAME not in existing_indexes:
    pc.create_index(INDEX_NAME, dimension=DIMS, metric='cosine', spec=spec)

    # Wait for the index to be initialized
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

for i in tqdm(range(0, len(data['id']), BATCH_SIZE)):
    # Find end of batch
    i_end = min(len(data['id']), i + BATCH_SIZE)

    # Create batch
    batch = {k: v[i:i_end] for k, v in data.items()}

    # Create embeddings
    chunks = [f'{x["title"]}: {x["content"]}' for x in batch["metadata"]]
    embeds = encoder.encode(chunks)

    # Ensure correct length
    assert len(embeds) == (i_end - i)

    # Upsert to Pinecone
    to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
    index.upsert(vectors=to_upsert)

def get_docs(query: str, top_k: int) -> list[str]:
    xq = encoder.encode(query)
    res = index.query(vector=xq.tolist(), top_k=top_k, include_metadata=True)
    return [x["metadata"]['content'] for x in res["matches"]]

def get_response(query: str, docs: list[str]) -> str:
    system_message = (
        "You are Anjibot, the AI course rep of 400 Level Computer Science department. You are always helpful, jovial, can be sarcastic but still sweet.\n"
        "Provide the answer to class-related queries using\n"
        "context provided below.\n"
        "If you don't the answer to the user's question based on your pretrained knowledge and the context provided, just direct the user to Anji the human course rep.\n"
        "Anji's phone number: 08145170886.\n\n"
        "CONTEXT:\n"
        "\n---\n".join(docs)
        )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content



def handle_query(user_query: str):

    # Get relevant documents
    docs = get_docs(user_query, top_k=5)

    # Generate and return response
    response = get_response(user_query, docs=docs)

    return response
