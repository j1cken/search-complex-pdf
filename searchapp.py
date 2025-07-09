from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from elasticsearch import Elasticsearch
import os
import sys
from google import genai
from PIL import Image
import time
import numpy as np
import base64
import io

app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv("elastic.env")
es_url = os.getenv("elastic_url")
es_api_key = os.getenv("elastic_api_key")
google_api_key = os.getenv("google_api_key")

google_model = "gemini-2.0-flash-lite"

client = genai.Client(api_key=google_api_key)
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
        session['search_str'] = query
        index_name = request.form.get('index')
        session['index_name'] = index_name
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
        # print(es_query)

        # ... bitvector search
        # query_vector = to_bit_vectors(create_col_pali_query_vectors(query))
        # es_query = {
        #     "_source": False,
        #     "query": {
        #         "script_score": {
        #             "query": {
        #                 "match_all": {}
        #             },
        #             "script": {
        #                 "source": "maxSimInvHamming(params.query_vector, 'col_pali_vectors')",
        #                 "params": {
        #                     "query_vector": query_vector
        #                 }
        #             }
        #         }
        #     },
        #     "size": 5
        # }

        results = es.search(index=index_name, body=es_query)
        #es_time = time.time() - start_time
        es_time = results['took'] / 1000

        # print(results["hits"]['hits'])
        images_base64 = [hit['_source']['image'] for hit in results['hits']['hits']]
        image_ids = [hit['_id'] for hit in results['hits']['hits']]
        session['image_ids'] = image_ids
        pdfs = [hit['_source']['pdf'] for hit in results['hits']['hits']]
        image_scores = [hit['_score'] for hit in results['hits']['hits']]
        # print(image_scores)

        # Return file paths and response times
        return jsonify(index=index_name, pdfs=pdfs, images=images_base64, img_scores=image_scores, es_time=es_time)

    return render_template('index.html', pdfs=[], images=[], response_text="", es_time=0, google_time=0, img_scores=[])

@app.route('/summarize', methods=['POST'])
def llm():
    query = session.get('search_str')
    num_docs = int(request.form.get('numdocs'))

    image_ids = session.get('image_ids', [])
    if not image_ids:
        return jsonify(response_text="No images found in session. Please perform a search first.", google_time=0, llm_model="N/A")

    # Fetch the images from Elasticsearch using the IDs
    images_base64 = []
    index_name = session.get('index_name')
    # Use Elasticsearch mget for bulk get
    mget_body = {
        "ids": image_ids[:num_docs]
    }
    docs = es.mget(index=index_name, body=mget_body)
    for doc in docs['docs']:
        if doc.get('found'):
            images_base64.append(doc['_source']['image'])

        # Measure Google Gemini query time
        google_time = 0
        rsptext = ""
        if llm:
            start_time = time.time()

            images = [Image.open(io.BytesIO(base64.b64decode(img_base64))) for img_base64 in images_base64]
            response = client.models.generate_content(
                model=google_model,
                contents=[images, query + " Answer the question in not more than 10 sentences. Result should be an easy-to-read paragraph. Only use the information on the images to answer the question and create a summary."]
            )
            rsptext = response.text
            google_time = time.time() - start_time

         # Return file paths and response times
        return jsonify(response_text=rsptext, google_time=google_time, llm_model=google_model)


if __name__ == '__main__':
    app.run(port=8000,debug=False)
