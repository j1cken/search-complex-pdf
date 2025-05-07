from dotenv import load_dotenv
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from elasticsearch import Elasticsearch
import os
import sys
import numpy as np

load_dotenv("elastic.env")
INDEX_NAME = os.getenv("index-name")
es_url = os.getenv("elastic_url")
es_api_key = os.getenv("elastic_api_key")

model_name = "vidore/colpali-v1.3"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="mps",  # "mps" for Apple Silicon, "cuda" if available, "cpu" otherwise
).eval()

col_pali_processor = ColPaliProcessor.from_pretrained(model_name)

def create_col_poli_image_vectors(image_path: str) -> list:
    batch_images = col_pali_processor.process_images([Image.open(image_path)]).to(model.device)
    with torch.no_grad():
        return model(**batch_images).tolist()[0]

def to_bit_vectors(embeddings: list) -> list:
    return [
        np.packbits(np.where(np.array(embedding) > 0, 1, 0))
        .astype(np.int8)
        .tobytes()
        .hex()
        for embedding in embeddings
    ]

# Index mapping
mappings = {
    "mappings": {
        "properties": {
            "col_pali_vectors": {
                "type": "rank_vectors",
                "element_type": "bit"
            }
        }
    }
}

# Connect to Elasticsearch
es = Elasticsearch(es_url, api_key=es_api_key)

# Check if the index already exists
if es.indices.exists(index=INDEX_NAME):
    print(f"Index '{INDEX_NAME}' already exists. Exiting script.")
    sys.exit()

# Create the index if it doesn't exist
es.indices.create(index=INDEX_NAME, body=mappings)

# Get the directory where this script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Define the output folder as the "static" subfolder in the script's directory
directory_path = os.path.join(script_directory, 'static')

for file_name in os.listdir(directory_path):
    # Construct full file path
    file_path = os.path.join(directory_path, file_name)
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        vectors = to_bit_vectors(create_col_poli_image_vectors(file_path))
        es.index(index=INDEX_NAME, id=file_path, document={"col_pali_vectors": vectors})
        # print(vectors)
