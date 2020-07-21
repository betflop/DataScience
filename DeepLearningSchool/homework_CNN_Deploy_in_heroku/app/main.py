import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from flask import Flask, render_template, request, redirect
from flask_dropzone import Dropzone
# from torchvision import models
# from PIL import Image
# from torchvision import transforms as T
# import torch
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')

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

def object_detection_api(img_path, path_new):

    classes = None
    with open(os.path.join(APP_STATIC, 'coco.names'),'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # read pre-trained model and config file
    net = cv2.dnn.readNet(os.path.join(APP_STATIC, "yolov3.weights"),os.path.join(APP_STATIC, "yolo.cfg"))

    # read input image
    image = cv2.imread(img_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        draw_bounding_box(image, classes, COLORS, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    plt.imshow(image)
    plt.savefig(path_new)

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

@app.route('/', methods=['POST', 'GET'])
def hello(name=None):
    if request.method == 'POST':
        f = request.files.get('file')
        path_file = os.path.join(app.config['UPLOADED_PATH'], f.filename)
        newfilename = 'detect2_{}'.format(f.filename)
        path_new = os.path.join(app.config['UPLOADED_PATH'], newfilename)
        f.save(path_file)
        object_detection_api(path_file, path_new)
        # newurl = '/result/{}'.format(newfilename)
        # return redirect(newurl, code=302)
    print(123)
    if request.method == 'GET':
        # if not 'model' in globals():
        #     print(321)
        #     global model
        #     model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        #     print('model_______')
        #     model.eval()
        #     print('model_______eval')
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
    