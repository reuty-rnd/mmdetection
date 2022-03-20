from torchvision.utils import save_image
import torch
from mmcv.runner.hooks import HOOKS, Hook

def all_convolutions_layer(model,convs=[]):
    if type(model) ==torch.nn.modules.conv.Conv2d:
        convs.append(model)
    else:
        try:
            for child in model.children():
                convs=all_convolutions_layer(child,convs)
        except:
            pass
    return convs

def norm_weights(mat):
    w_min=mat.min()
    w_max=mat.max()
    mat= (mat-w_min)/(w_max-w_min)
    return mat

def show_filters(model,sfx=''):
    all_convs= all_convolutions_layer(model)
    first_l_w=all_convs[0].weight
    deep_l_w=all_convs[6].weight
    save_image(first_l_w,'./hackATR_results/layer1{}.png'.format(sfx))
    save_image(deep_l_w[:,0:3,:],'./hackATR_results/layer6{}.png'.format(sfx))
    save_image(deep_l_w.transpose(1,2),'./hackATR_results/layer6_transpose{}.png'.format(sfx))
    
    # save_image(norm_weights(first_l_w),'layer1{}.png'.format(sfx))
    # save_image(norm_weights(deep_l_w[:,0:3,:]),'layer6{}.png'.format(sfx))
    # save_image(norm_weights(deep_l_w.transpose(1,2)),'layer6_transpose{}.png'.format(sfx))
    
@HOOKS.register_module()
class ShowFiltersHook(Hook):


    def _all_convolutions_layer(self, model,convs=[]):
        if type(model) ==torch.nn.modules.conv.Conv2d:
            convs.append(model)
        else:
            try:
                for child in model.children():
                    convs=self._all_convolutions_layer(child,convs)
            except:
                pass
        return convs

    # def _norm_weights(self, mat):
    #     w_min=mat.min()
    #     w_max=mat.max()
    #     mat= (mat-w_min)/(w_max-w_min)
    #     return mat

    def _show_filters(self,model,sfx=''):
        all_convs= self._all_convolutions_layer(model)
        first_l_w=all_convs[0].weight
        deep_l_w=all_convs[6].weight
        save_image(first_l_w,'./hackATR_results/layer1{}.png'.format(sfx))
        save_image(deep_l_w[:,0:3,:],'./hackATR_results/layer6{}.png'.format(sfx))
        save_image(deep_l_w.transpose(1,2),'./hackATR_results/layer6_transpose{}.png'.format(sfx))
        
        # save_image(norm_weights(first_l_w),'layer1{}.png'.format(sfx))
        # save_image(norm_weights(deep_l_w[:,0:3,:]),'layer6{}.png'.format(sfx))
        # save_image(norm_weights(deep_l_w.transpose(1,2)),'layer6_transpose{}.png'.format(sfx))
        

   

    def before_run(self, runner):
        model = runner.model
        self._show_filters(model,'before_init')


    def after_run(self, runner):
        model = runner.model
        self._show_filters(model,'after_init')


# def before_train_epoch(self, runner):
#     """Check whether the training dataset is compatible with head.

#     Args:
#         runner (obj:`EpochBasedRunner`): Epoch based Runner.
#     """
#     self._check_head(runner)

# def before_val_epoch(self, runner):
#     """Check whether the dataset in val epoch is compatible with head.

#     Args:
#         runner (obj:`EpochBasedRunner`): Epoch based Runner.
#     """
#     self._check_head(runner)
