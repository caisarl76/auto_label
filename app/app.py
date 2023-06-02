import json
from flask import Flask
from flask import request, Response
from flask_cors import CORS
import cv2
import numpy as np
import io
# import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, FCN_ResNet50_Weights, FCN_ResNet101_Weights
import copy
from PIL import Image, ImageOps
from io import BytesIO
import base64

#cls_list
cls_list = ['dog','cat', 'bird', 'cow', 'horse']

tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def get_obj(img, model_type, model_path):
    # os.makedirs('./output', exist_ok=True)
    
    assert model_type in ['resnet50', 'resnet101']
    if model_type == 'resnet50':
        weights = FCN_ResNet50_Weights.DEFAULT
        transforms = weights.transforms(resize_size=None)
        model = fcn_resnet50(weights=weights, progress=False)
        
    elif model_type == 'resnet101':
        weights = FCN_ResNet101_Weights.DEFAULT
        transforms = weights.transforms(resize_size=None)
        model = fcn_resnet101(weights=weights, progress=False)
        
    else:
        return
    model = model.eval()
    
    batch = torch.stack([transforms(img)])
    output = model(batch)['out']

    sem_class_to_idx = {cls:idx for (idx, cls) in enumerate(weights.meta["categories"])}
    
    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    
    boolean_masks = [(normalized_masks.argmax(1) == sem_class_to_idx[cls]).type(torch.uint8) for cls in cls_list]
    
    contour_list = []
    obj_list = []
    obj_dict = {
        "attributes":[],
        "type":"polygon",
        "points":[]
    }
    
    for mask in boolean_masks:
        mask = mask.numpy().transpose(1, 2, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list.append(contours)
        for cont in contours:
            new_obj = copy.deepcopy(obj_dict)
            if len(cont)>3:
                for point in cont:
                    x, y = point[0]
                    new_obj['points'].append(int(x))
                    new_obj['points'].append(int(y))
                obj_list.append(new_obj)
    # image = np.array(img).astype(np.uint8)
    # image = cv2.drawContours(image, contour_list[0], -1, (0, 255, 0), 3)
    # cv2.imwrite('image.png', image)
    json_obj = {'object':obj_list}
    return json_obj

def prepare_data(request):
    if request.mimetype == 'application/json':
        data = request.json
        model_type = data['model_type']
        model_path = data['model_path'] 
        
        info, image = data['image'].split(';')
        image = base64.b64decode(image.split('base64,')[-1])
        image = io.BytesIO(image)
        image = Image.open(image)
        
        data = {
            'image':[image],
            'model_type': [model_type],
            'model_path':[model_path]
        }
        
        print(f'Json data Found: {data}', flush=True)
    elif request.mimetype == 'multipart/form-data':
        files = dict(request.files.lists())
        data = request.form.to_dict(flat=False)
        print('Parsing Images...', flush=True)
        data['image'] = read_images(files)
        # print(f"Image data found: {[img.shape for img in data['image']]}", flush=True)
    else:
        raise TypeError('Invalid Content-Type: %s' % request.mimetype)
    return data

def read_images(files):
    if 'image' not in files:
        raise NotImplementedError(f"File keys other than \"image\" are not allowed, but found: {list(files.keys())}")
    # img_batch = [np.array(ImageOps.exif_transpose(Image.open(BytesIO(stream.read())))) for stream in files['image']]
    img_batch = [ImageOps.exif_transpose(Image.open(BytesIO(stream.read())).convert('RGB')) for stream in files['image']]
    return img_batch


app = Flask('seg-flask')
app.config['JSON_AS_ASCII'] = False
CORS(app,
    resources={'*': {'origins': '*'}},
    supports_credentials=True,
    methods=['*'],
    allow_headers=['*']
)

@app.route('/', methods=['POST'])
def home():
    data = prepare_data(request)
    img = data.get('image')[0]
    model_type = data.get('model_type')[0]
    model_path = data.get('model_path')[0]
    # 코드
    json_obj = get_obj(img, model_type, model_path)

    return json_obj

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10051, debug=True)
    