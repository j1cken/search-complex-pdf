from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from elasticsearch import Elasticsearch
import os
import sys
from PIL import Image
import time
import numpy as np
import base64
import io

app = Flask(__name__)

load_dotenv("elastic.env")
es_url = os.getenv("elastic_url")
es_api_key = os.getenv("elastic_api_key")

es = Elasticsearch(es_url, api_key=es_api_key, verify_certs=True)

model_name = "vidore/colpali-v1.3"
model = ColPali.from_pretrained(
    "vidore/colpali-v1.3",
    torch_dtype=torch.float32,
    device_map="mps",  # "mps" for Apple Silicon, "cuda" if available, "cpu" otherwise
).eval()

col_pali_processor = ColPaliProcessor.from_pretrained(model_name)

def create_col_pali_query_vectors(query: str) -> list:
    queries = col_pali_processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        return model(**queries).tolist()[0]

def to_bit_vectors(embeddings: list) -> list:
    return [
        np.packbits(np.where(np.array(embedding) > 0, 1, 0))
        .astype(np.int8)
        .tobytes()
        .hex()
        for embedding in embeddings
    ]

@app.route('/indices', methods=['GET'])
def get_es_indices():
    # Get all indices from Elasticsearch
    es_indices_info = es.indices.get_alias(index='*', expand_wildcards='all')
    # print(es_indices_info.keys())
    es_indices = list(es_indices_info.keys())
    # print(es_indices)
    return jsonify(es_indices)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('search_string')
        llm = request.form.get('llm')
        index_name = request.form.get('index')
        # print(index_name)

        # Measure Elasticsearch query time
        start_time = time.time()
        es_query = {
            "_source": ["image", "pdf"],
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "maxSimDotProduct(params.query_vector, 'col_pali_vectors')",
                        "params": {"query_vector": create_col_pali_query_vectors(query)},
                    },
                }
            },
            "size": 5,
        }

        results = es.search(index=index_name, body=es_query)
        #es_time = time.time() - start_time
        es_time = results['took'] / 1000

        # print(results["hits"]['hits'])
        images_base64 = [hit['_source']['image'] for hit in results['hits']['hits']]
        pdfs = [hit['_source']['pdf'] for hit in results['hits']['hits']]
        image_scores = [hit['_score'] for hit in results['hits']['hits']]
        # print(image_scores)

        # Measure Google Gemini query time
        google_time = 0
        rsptext = ""
        if llm:
            start_time = time.time()

            # images = [Image.open(io.BytesIO(base64.b64decode(img_base64))) for img_base64 in images_base64]

            # Send images to Ollama chat API to generate a summary
            import requests

            ollama_url = "http://localhost:11434/api/generate"
            ollama_model = "llava:13b"  # or another multimodal model available in Ollama

            ollama_payload = {
                "model": ollama_model,
                "role": "user",
                "prompt": f"Answer this question: {query}. Use the images only.",
                "stream": False,
                "images": images_base64[:1],
            }

            try:
                ollama_response = requests.post(ollama_url, json=ollama_payload, timeout=60)
                ollama_response.raise_for_status()
                # print(ollama_response.json())
                ollama_summary = ollama_response.json().get("response", {})
            except Exception as e:
                ollama_summary = f"Ollama error: {e}"

            rsptext = ollama_summary
            google_time = time.time() - start_time

        # Return file paths and response times
        return jsonify(index=index_name, pdfs=pdfs, images=images_base64, response_text=rsptext, es_time=es_time, google_time=google_time, img_scores=image_scores, llm_model=ollama_model)

    return render_template('index.html', pdfs=[], images=[], response_text="", es_time=0, google_time=0, img_scores=[])

if __name__ == '__main__':
    app.run(port=8000,debug=False)
