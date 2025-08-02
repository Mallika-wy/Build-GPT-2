from torch import dtype

from config import PretrainedConfig
from utils import *


class GPTPreTrainedModel(nn.Module):

  def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
    super().__init__()
    self.config = config
    self.name_or_path = config.name_or_path

  def init_weights(self):
    # Initialize weights
    # apply 方法会递归地将 _init_weights 方法应用到模型的每个子模块上
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """ Initialize the weights """
    # 对于不同的层，采用不同的初始化方式
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # 对于线性层和嵌入层，使用正态分布初始化
      # 这里的std是标准差，mean是均值
      # 这样有助于模型训练权重分布合理，避免梯度爆炸或者消失
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      # 对于层归一化层，初始化权重为1，偏置为0
      # 这样保证归一化层在训练初期的输出稳定
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @property
  def dtype(self) -> dtype:
    return get_parameter_dtype(self)

