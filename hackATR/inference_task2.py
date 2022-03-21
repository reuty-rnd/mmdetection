from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch
from torchvision.utils import save_image 
import cv2

from show_objectness import SaveOutput, show_objectnesses

def get_layer_modules(model):
    return dict([*model.named_modules()])

config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
#config_file ='./hackATR/config_hackATR.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

layers_name_dict=get_layer_modules(model)
objectness_layer=layers_name_dict['rpn_head.rpn_cls']
save_output = SaveOutput()
handle=objectness_layer.register_forward_hook(save_output)

# test a single image
img = './demo/demo.jpg'
result = inference_detector(model, img)

#save the objecness results
#save_image(save_output.outputs[0][0,:,:,:],'./hackATR_results/task2_objectness.png')
img=cv2.imread(img)
show_objectnesses(img,save_output.outputs)

# show the results
#show_result_pyplot(model, img, result)