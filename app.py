from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
import pytesseract
import torch
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, ViTForImageClassification, ViTConfig, ViTImageProcessor
from scipy.special import softmax
import numpy as np
import cv2
from torchvision import transforms

app = Flask(__name__)

# Directory setup
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'txt'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_final(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, im = cv2.threshold(im, 240, 255, cv2.THRESH_BINARY)  # Apply binary thresholding
    return im

# Function to extract text from an image
def extract_text(image_path):
    custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist= 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '"
    img = cv2.imread(image_path)
    img = preprocess_final(img)
    text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
    return text.replace('\n', ' ').lower()  # Convert extracted text to lower case

# Function to transform image with processor
def transform_image_with_processor(image, processor):
    return processor(images=image, return_tensors="pt")['pixel_values']

# Load RoBERTa tokenizer (common for both models)
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dataset_type = request.form['dataset']

    # Load appropriate models based on dataset selection
    if dataset_type == 'public':
        text_model = TFRobertaForSequenceClassification.from_pretrained('C:/Users/israt/Desktop/website_2/public_roberta-cyberbullying-classifier')
        image_model_path = 'C:/Users/israt/Desktop/website_2/public_vit_model.pth'
    else:
        text_model = TFRobertaForSequenceClassification.from_pretrained('C:/Users/israt/Desktop/website_2/private_roberta-cyberbullying-classifier')
        image_model_path = 'C:/Users/israt/Desktop/website_2/private_vit_model.pth'

    # Load ViT model and processor
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=4)
    vit_model.load_state_dict(torch.load(image_model_path, map_location=torch.device('cpu')))
    vit_model.eval()

    text_label = "No text prediction"
    image_label = "No image prediction"
    extracted_text = "No text extracted"
    fusion_message = "Pending analysis..."  # Initialize fusion message

    # Process Text Input
    text = request.form.get('singleText')
    text_class = None  # Initialize text_class variable
    if text:
        inputs = roberta_tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=512, 
            padding='max_length', truncation=True, return_tensors="tf"
        )
        roberta_prediction = text_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        roberta_probs = softmax(roberta_prediction[0].numpy(), axis=1)
        text_class = np.argmax(roberta_probs, axis=1)[0]
        text_label = f"Text class label: {text_class}"

    # Process Image Input
    image_file = request.files.get('image')
    image_class = None  # Initialize image_class variable
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # Extract text from the image
        extracted_text = extract_text(image_path)

        # Process extracted text
        if extracted_text.strip():
            inputs = roberta_tokenizer.encode_plus(
                extracted_text, add_special_tokens=True, max_length=512,
                padding='max_length', truncation=True, return_tensors="tf"
            )
            roberta_prediction = text_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            roberta_probs = softmax(roberta_prediction[0].numpy(), axis=1)
            text_class = np.argmax(roberta_probs, axis=1)[0]
            text_label = f"Extracted text class label: {text_class}"

        # Process Image with ViT model
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform_image_with_processor(image, vit_processor)
        with torch.no_grad():
            outputs = vit_model(input_tensor)
            vit_probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            image_class = torch.argmax(vit_probs, dim=1).item()
            image_label = f"Image class label: {image_class}"

    # Fusion Logic
    if text_class is not None and image_class is not None:
        if text_class == image_class:
            if text_class == 0:
                fusion_message = f"Input does not contain any Cyber-bullying.  {text_label}  and  {image_label}"
            else:
                fusion_message = f"Input contains  cyberbullying of class label: {text_class}."
        else:
            fusion_message = f"Input contains cyberbullying.   {text_label}  and  {image_label}"
    else:
        fusion_message = "This is not Multi-Modal Data!"


    results = {
        'extracted_text': extracted_text,
        'text_label': text_label,
        'image_label': image_label,
        'fusion_message': fusion_message  # Add the fusion result to the output
    }

    return render_template('results.html', results=results)



if __name__ == '__main__':
    app.run(debug=True)
