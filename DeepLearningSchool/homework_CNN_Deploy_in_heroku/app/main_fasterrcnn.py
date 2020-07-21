from torchvision import models
from PIL import Image
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
import os
import json
from flask import Flask, render_template, request, redirect
from flask_dropzone import Dropzone
import torch

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=2,
)
dropzone = Dropzone(app)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# model = None

# @app.before_first_request
# def initialize():
#     print('Hello_______')
#     if not 'model' in globals():
#         global model
#         model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#         print('model_______')
#         model.eval()
#         print('model_______eval')

def get_prediction(img_path, threshold):
    print('get_prediction')
    with torch.no_grad():
        print('torch.no_grad()')
        img = Image.open(img_path) # Load the image
        img.resize((20, 10))
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img) # Apply the transform to the image
        print('transform')
        model.eval()
        print('model.eval')
        pred = model([img]) # Pass the image to the model
        print('predict')
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        print('pred_class')
        return pred_boxes, pred_class

def object_detection_api(img_path, path_new, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
        if pred_cls[i] != 'person':
            continue
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    plt.imshow(img)
    plt.savefig(path_new)
    print('path_new - {}'.format(path_new))

@app.route('/', methods=['POST', 'GET'])
def hello(name=None):
    if request.method == 'POST':
        f = request.files.get('file')
        path_file = os.path.join(app.config['UPLOADED_PATH'], f.filename)
        newfilename = 'detect2_{}'.format(f.filename)
        path_new = os.path.join(app.config['UPLOADED_PATH'], newfilename)
        f.save(path_file)
        object_detection_api(path_file, path_new)
        newurl = '/result/{}'.format(newfilename)
        return redirect(newurl, code=302)
    print(123)
    if request.method == 'GET':
        if not 'model' in globals():
            print(321)
            global model
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            print('model_______')
            model.eval()
            print('model_______eval')
        return render_template('hello.html')

@app.route('/result/<name>', methods=['POST', 'GET'])
def result(name=None):
    return render_template('result.html', name=name)

# if __name__ == '__main__':
# #     app.run(host='0.0.0.0')
#     print('Hello_______')
    
#     if model is None:
#         model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#         print('model_______')
#         model.eval()
#         print('model_______eval')
    