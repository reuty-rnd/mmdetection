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

    
@HOOKS.register_module()
class ShowObjectnessHook(Hook):


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

   

    def after_iter(self, runner):
        model = runner.model
        self._all_convolutions_layer(model)


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
