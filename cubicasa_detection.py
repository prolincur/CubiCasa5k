# docker run -d --rm --init --gpus all --ipc=host --publish 1111:1111 -e NVIDIA_VISIBLE_DEVICES=0 cubi_localonly:0.1.0

import os
from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from floortrans.models import get_model
from floortrans.plotting import polygons_to_image, discrete_cmap
discrete_cmap()
from floortrans.post_prosessing import split_prediction, get_polygons
import cv2
import logging
from io import BytesIO
import base64
from datetime import datetime
from tracing_util import raster_to_vector

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('flask')
logger.setLevel(logging.INFO)  # Set the logger to capture info level logs
handler = logging.StreamHandler()  # Creates a stream handler that logs to stdout
handler.setLevel(logging.INFO)  # Ensure the handler captures info level logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)  # Adds the handler to the Flask logger

OUTPUT_DIR = "output_images" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def img_to_base64(img_array, format="PNG"):
    """
    Converts an image array into a base64 encoded string.
    """
    img = BytesIO()
    plt.imsave(img, img_array, format=format)
    img.seek(0)  # Go to the beginning of the BytesIO buffer
    base64_img = base64.b64encode(img.getvalue()).decode("utf-8")
    return base64_img

@app.route("/polygon-gen")
def health_check(): # initialised for AWS ECS health checks a '/' location
    logger.info("Received health check request")
    return jsonify({"status": "ok"}), 200

@app.route("/polygon-gen/simplify_floorplan", methods=['POST'])
def process_image():
    logger.info("Received request to process image")

    if 'file' not in request.files:
        logger.error("No file provided in the request")
        return jsonify({"error": "no file provided"}), 400

    file = request.files['file']
    
    if file.filename == '':
        logger.error("No file selected for uploading")
        return jsonify({"error": "no file selected for uploading"}), 400
    
    # Setup Model
    model = get_model('hg_furukawa_original', 51)

    n_classes = 44
    split = [21, 12, 11]

    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    print("new version")

    checkpoint = torch.load('model/model_best_val_loss_var.pkl', map_location='cpu') # This is for local where the model is in the same container
    #checkpoint = torch.load('/model/model/model_best_val_loss_var.pkl', map_location='cpu') # This is for Azure where the models are in a separate volume

    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    logger.info("Model loaded.")

    # Directly read the uploaded file into a numpy array
    nparr = np.fromstring(file.read(), np.uint8)
    fplan_orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    fplan = cv2.cvtColor(fplan_orig, cv2.COLOR_BGR2RGB)  # correct color channels
    fplan = np.moveaxis(fplan, -1, 0)  # correct the dimension order
    fplan = 2 * (fplan / 255.0) - 1  # Normalize values to range -1 and 1
    image = torch.FloatTensor(fplan)  # Convert NumPy array to PyTorch tensor
    image = image.unsqueeze(0)  # add extra dim 


    prediction = model(image)  # pass the tensor to the model

    with torch.no_grad():
        height, width = image.shape[2], image.shape[3]
        img_size = (height, width)

        # Extracting room and icon predictions
        rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
        rooms_pred = np.argmax(rooms_pred, axis=0) # rooms prediction

        icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
        icons_pred = np.argmax(icons_pred, axis=0) # icons prediction

    # Process the prediction to get heatmaps, rooms, and icons
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

    pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width) # post process room and icon 

    # Save the images to the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    output_files = {
        "pol_room_seg": os.path.join(OUTPUT_DIR, f"{timestamp}_walls_segmented.png"),
        "vector_output_path": os.path.join(OUTPUT_DIR, f"{timestamp}_vector_output_path.geojson")
    }

    # Save the images
    plt.imsave(output_files["pol_room_seg"], pol_room_seg, format="PNG")

    # vectorization
    logger.info("Starting vectorization of the post-processed image")
    raster_to_vector(output_files["pol_room_seg"], output_files["vector_output_path"])
    logger.info("Vectorization completed")

    logger.info(f"Saved output images to {OUTPUT_DIR}")

    # Return paths to the saved images
    return jsonify({
        "output_files": output_files,
    })


if __name__ == "__main__":
    logger.info("Starting Flask application on port 1111")
    app.run(host='0.0.0.0', port=1111, debug=True)
