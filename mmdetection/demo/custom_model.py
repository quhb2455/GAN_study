from ..builder import NECKS
from torch import nn

@NECKS.register_module()
class CUSTOM_MODEL(nn.Module) :
    def __init__(self,
                 in_channel,
                 out_channel,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False):
        pass

    def forward(self, inputs):
        pass