import torch
from matplotlib import pyplot as plt
import cv2


class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []


def show_objectnesses(img, featuremaps):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mmcv turns img into bgr
    shape_img=(img.shape[1],img.shape[0])

    for scale in range(0,5):
        fig, axes = plt.subplots(2, 2, figsize= (16,9))
        axes[0,0].imshow(img)
        axes[0,0].set_title('Original image')

        # axes[0,1].imshow(img, alpha=0.9)
        featuremap = featuremaps[scale][0][0].cpu().numpy()
        axes[0,1].imshow(cv2.resize(featuremap, shape_img), alpha=0.5)
        axes[0,1].set_title('Scale: '+str(scale) + ', Anchor: 0')

        # axes[1, 0].imshow(img, alpha=0.9)
        featuremap = featuremaps[scale][0][1].cpu().numpy()
        axes[1, 0].imshow(cv2.resize(featuremap, shape_img), alpha=0.5)
        axes[1, 0].set_title('Scale: ' + str(scale) + ', Anchor: 1')

        # axes[1,1].imshow(img, alpha=0.9)
        featuremap = featuremaps[scale][0][2].cpu().numpy()
        axes[1,1].imshow(cv2.resize(featuremap, shape_img), alpha=0.5)
        axes[1,1].set_title('Scale: '+str(scale) + ', Anchor: 2')

        plt.savefig('./hackATR_results/feature_results_of_scale_' + str(scale)+'.png', bbox_inches='tight')