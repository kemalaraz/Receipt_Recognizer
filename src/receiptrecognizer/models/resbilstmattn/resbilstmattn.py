"""
Copyright (c) 2019-present NAVER Corp.

Model taken from https://github.com/clovaai/deep-text-recognition-benchmark
"""

import torch.nn as nn

from ..utils import TPS_SpatialTransformerNetwork
from ..backbones import RCNN_FeatureExtractor, ResNet_FeatureExtractor
from .blocks import BidirectionalLSTM, Attention


class ReNBAtt(nn.Module):

    __backbones__ = {
                        "rcnn": RCNN_FeatureExtractor,
                        "resnet": ResNet_FeatureExtractor
                    }

    def __init__(self, num_class:int, hidden_size:int = 256, imgH:int = 32, imgW:int = 100, num_fiducial:int = 20,
                    input_channel:int = 1, output_channel:int = 512, backbone:str = "resnet", batch_max_len:int = 25):
        super(ReNBAtt, self).__init__()
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.imgH, self.imgW = imgH, imgW
        self.num_fiducial = num_fiducial
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.backbone = backbone
        self.batch_max_len = batch_max_len


        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
                F=self.num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)

        """ FeatureExtraction """
        assert self.backbone in ReNBAtt.__backbones__.keys(), ("Given backbone is not in the backbone list"
                                                            f"Supported backbones : {list(ReNBAtt.__backbones__.keys())}")
        self.FeatureExtraction = ReNBAtt.__backbones__[self.backbone](input_channel, output_channel)

        self.FeatureExtraction_output = self.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size, self.hidden_size),
            BidirectionalLSTM(self.hidden_size, self.hidden_size, self.hidden_size))
        self.SequenceModeling_output = self.hidden_size

        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, self.hidden_size, self.num_class)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)


        """ Prediction stage """
        # if self.stages['Pred'] == 'CTC':
        #     prediction = self.Prediction(contextual_feature.contiguous())
        # else:
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length = self.batch_max_len)

        return prediction
