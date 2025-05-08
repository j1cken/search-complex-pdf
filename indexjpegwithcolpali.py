from dotenv import load_dotenv
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import os
import io
import sys
import numpy as np
import fitz
import tempfile
import uuid
import base64

load_dotenv("elastic.env")
# INDEX_NAME = os.getenv("index-name")
es_url = os.getenv("elastic_url")
es_api_key = os.getenv("elastic_api_key")

# Connect to Elasticsearch
es = Elasticsearch(es_url, api_key=es_api_key)

model_name = "vidore/colpali-v1.3"

def is_valid_pdf(pdf_path):
    """Check if the provided path is a valid PDF file."""
    if not os.path.isfile(pdf_path):
        print(f"Error: The path {pdf_path} does not exist or is not a valid file.")
        return False
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: The file at {pdf_path} is not a PDF (it doesn't have a .pdf extension).")
        return False
    try:
        # Try to open the PDF to check if it's a valid PDF file
        fitz.open(pdf_path)
    except Exception as e:
        print(f"Error: The file at {pdf_path} is not a valid PDF. {e}")
        return False
    return True

def pdf_to_jpeg(pdf_path, output_folder):
    """Convert each page of a PDF to a JPEG image and save them in the output folder."""
    # Ensure the output directory exists and is empty
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    num_pages = pdf_document.page_count
    for page_number in range(num_pages):
        # Render page to an image
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        # Save the image as JPEG
        output_path = os.path.join(output_folder, f"page_{page_number + 1}.jpeg")
        img.save(output_path, "JPEG")
    # Close the PDF document
    pdf_document.close()

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
            },
            "image": {
                "type": "binary"
            },
            "pdf": {
                "type": "text"
            }
        }
    }
}

def index_it(index_name, pdf_file, output_folder):
    for file_name in os.listdir(output_folder):
        # Construct full file path
        file_path = os.path.join(output_folder, file_name)
        # Check if it's an image file with allowed extension
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            if pdf_file=="":
                pdf_name = file_name
            else:
                pdf_name = pdf_file
            # Convert image to base64 representation
            with open(file_path, "rb") as img_file:
                imageb64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                vectors = to_bit_vectors(create_col_poli_image_vectors(file_path))
                # es.index(index=INDEX_NAME, id=file_name, document={"col_pali_vectors": vectors, "image": imageb64})
                yield {"_index": index_name, "col_pali_vectors": vectors, "image": imageb64, "pdf": pdf_name}
                # print(vectors)

def main():
    # Get the command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <index_name> <pdf_path>")
        return

    INDEX_NAME = sys.argv[1]
    pdf_path = sys.argv[2]

    # Check if the index already exists
    if es.indices.exists(index=INDEX_NAME):
        print(f"WARNING: Index '{INDEX_NAME}' already exists.")
    else:
        # Create the index if it doesn't exist
        es.indices.create(index=INDEX_NAME, body=mappings)

    # Recursively walk through directories to find all PDF files
    plain_imgs_dirs = set()
    for root, dirs, files in os.walk(pdf_path):
        for pdf_file in files:
            if pdf_file.lower().endswith('.pdf'):
                pdf_fqpath = os.path.join(root, pdf_file)
                # Use a temporary directory as output_folder
                output_folder = os.path.join(tempfile.gettempdir(), 'pdf_images_' + str(uuid.uuid4()))
                # Convert PDF to JPEG images
                if not is_valid_pdf(pdf_fqpath):
                    print(f"Invalid PDF file: {pdf_path}/{pdf_file}")
                    continue
                pdf_to_jpeg(pdf_fqpath, output_folder)

                helpers.bulk(es, index_it(INDEX_NAME, pdf_file, output_folder))

            if pdf_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                plain_imgs_dirs.add(root)

    for dir in plain_imgs_dirs:
        helpers.bulk(es, index_it(INDEX_NAME, "", dir))

if __name__ == "__main__":
    main()
