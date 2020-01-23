#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torchvision


class BCNN_Normalized(torch.nn.Module):
    """
    B-CNN
    The structure of B-CNN is as follows:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1 ->
        conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2 ->
        conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> pool3 ->
        conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> pool4 ->
        conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) -> relu ->
        bilinear pooling -> signed square root normalize -> L2 normalize ->
        fc(n_classes)
    The network input 3*448*448 image
    The output of last convolution layer is 512 * 28 * 28

    A two step training procedure is adopted as the paper illustrated:
        in the first step, using pretrained VGG-16 to train the last fc only while freezing the previous layers
        in the second step, using unpretrained VGG-16 to train the entire network using back propagation
    """
    def __init__(self, n_classes=200, pretrained=True):
        """
        Declare all layers needed

            Arguments
                pretrained  [bool]  whether the VGG16 needs to be pretrained, default: True
                                    specify as True in first step and False in second step
        """
        super().__init__()
        self._pretrained = pretrained
        self._n_classes = n_classes
        # Convolution and pooling layers of VGG-16
        self.features = torchvision.models.vgg16(pretrained=self._pretrained).features
        # Remove Pool5, using the outputs of the last convolutional layer with non-linearities
        # as feature extractors, according to the original CVPR paper
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1])
        # Linear classifier
        # 2 stages training procedure, train the last fc only first according to the original paper
        self.fc = torch.nn.Linear(512**2, self._n_classes, bias=False)

        if self._pretrained:
            # Freeze all previous layers
            for param in self.features.parameters():
                param.requires_grad = False
            # Initialize the fc layer
            torch.nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        """
        Forward pass of the network

        Arguments:
            X [torch.Tensor] shape is N * 3 * 448 * 448

        Return:
            X [torch.Tensor] shape is N * 200
        """
        N = X.size()[0]  # N is the batch size
        assert X.size() == (N, 3, 448, 448), 'The image size should be 3x448x448(CxHxW)'
        X = self.features(X)  # extract features using pretrained VGG-16
        assert X.size() == (N, 512, 28, 28), 'The feature size should be 512x28x28(CxHxW)'
        X = X.view(N, 512, 28**2)
        # Bilinear, bmm is a batch matrix-matrix product of product,
        # corresponding to x = A^T*B in original paper
        # (here, A and B are both (N, 28**2, 512))
        # (here, the first param in bmm is A^T, the second param is B)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)
        assert X.size() == (N, 512, 512), 'bilinear op product wrong result'
        # reshape to get bilinear vector
        X = X.view(N, 512**2)
        # since X comes from relu5_3 of VGG, it is nonnegative elementwise
        X = torch.sqrt(X + 1e-5)
        # perform L2 normalize over X
        X = torch.nn.functional.normalize(X)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        X = self.fc(X)
        X = X * 30
        assert X.size() == (N, self._n_classes), 'Wrong fc op output'
        return X

class BCNN(torch.nn.Module):
    """
    BCNN

    The structure of BCNN is as follows:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1(64*224*224)
    ->  conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2(128*112*112)
    ->  conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> pool3(256*56*56)
    ->  conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> pool4(512*28*28)
    ->  conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) -> relu(512*28*28)
    ->  bilinear pooling(512**2)
    ->  fc(n_classes)

    The network input 3 * 448 * 448 image
    The output of last convolution layer is 512 * 14 * 14

    Extends:
        torch.nn.Module
    """
    def __init__(self, pretrained=True, n_classes=200):
        super().__init__()
        self._pretrained = pretrained
        self._n_classes = n_classes
        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)
        self.features = torch.nn.Sequential(*list(vgg16.features.children())[:-1])
        self.fc = torch.nn.Linear(512**2, self._n_classes)

        if self._pretrained:
            # Freeze all layer in self.feature
            for params in self.features.parameters():
                params.requires_grad = False
            # Init the fc layer
            torch.nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        """
        Forward pass of the network

        Arguments:
            x [torch.Tensor] -- shape is (N, 3, 448, 448)

        Return:
            x [torch.Tensor] -- shape is (N, n_classes)
        """
        N = x.size(0)
        # assert x.size() == (N, 3, 448, 448), 'The image size should be 3 x 448 x 448'
        x = self.features(x)
        bp_output = self.bilinear_pool(x)
        x = self.fc(bp_output)
        assert x.size() == (N, self._n_classes)
        return x

    @staticmethod
    def bilinear_pool(x):
        N, ch, h, w = x.shape
        # assert x.size() == (N, 512, 28, 28)
        x = x.view(N, 512, h*w)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (h * w)
        x = x.view(N, 512**2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        assert x.size() == (N, 512**2)
        return x