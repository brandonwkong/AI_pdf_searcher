import os
import io
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import fitz  

app = Flask(__name__)
CORS(app)

# Load the CLIP model and processor from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_images_with_coordinates(pdf_path: str):
    image_embeddings = []
    image_coordinates = []  
    images = []

    if not os.path.exists(pdf_path):
        return None, f"Error: File does not exist at {pdf_path}"
    #Open pdf with pdfplumber to get coords
    with pdfplumber.open(pdf_path) as pdf:
        for page_no in range(len(pdf.pages)):
            page = pdf.pages[page_no]
            images_in_page = page.images
            
            page_height = page.height
            
            if images_in_page:
                for i, image in enumerate(images_in_page):
                    x0 = image['x0']
                    y0 = image['y0']
                    x1 = image['x1']
                    y1 = image['y1']

                    coords = {
                        "page": page_no + 1,
                        "index": i,
                        "x0": x0,
                        "y0": page_height - y1,  
                        "x1": x1,
                        "y1": page_height - y0,  
                        "width": 612,
                        "height": 792,
                    }
                    image_coordinates.append(coords)

    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]  
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            images.append(image_bytes)

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Get image embeddings
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            image_embeddings.append(normalized_embeddings)

    return image_embeddings, image_coordinates 

# Helper function to find the best match
def find_closest_matches(user_input: str, image_embeddings: list, image_coordinates: list, threshold: float = 0.25):
    text_inputs = processor(text=user_input, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

    matches = []  

    for index, image_embedding in enumerate(image_embeddings):
        similarity = torch.cosine_similarity(image_embedding, text_features).item()

        if similarity >= threshold:
            matches.append({
                "index": index,
                "coordinates": image_coordinates[index]  # Store the coordinates of the match
            })

    if matches:
        return matches  # Return the list of matches
    else:
        return {"message": "No close matches found"}

# Flask route for extracting images and finding the best match
@app.route('/extract-images', methods=['POST'])
def extract_images_route():
    if 'pdf' not in request.files or 'user_input' not in request.form:
        return jsonify({"error": "No PDF file or user input provided."}), 400

    pdf_file = request.files['pdf']
    user_input = request.form['user_input']

    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({"error": "Unsupported file type. Please upload a PDF."}), 415

    # Save the PDF file temporarily
    temp_pdf_path = os.path.join("temp", pdf_file.filename)
    os.makedirs("temp", exist_ok=True)
    pdf_file.save(temp_pdf_path)

    image_embeddings, image_coordinates = extract_images_with_coordinates(temp_pdf_path)

    if image_embeddings is None:  
        return jsonify({"error": image_coordinates}), 500  

    result = find_closest_matches(user_input, image_embeddings, image_coordinates)

    # Log the result being returned
    #print("Response being sent back:", result)  # Log the response content
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
