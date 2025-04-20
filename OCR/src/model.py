import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Bi-directional LSTM for text recognition.
    Args:
        nIn (int): number of input units
        nHidden (int): number of hidden units
        nOut (int): number of output units
    """
    def __init__(self, nIn, nHidden, nOut):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        rnnOut, _ = self.rnn(input)
        T, b, c = rnnOut.size()
        rnnOut = rnnOut.view(T * b, c)
        output = self.embedding(rnnOut)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    """
    CRNN model for text recognition.
    Args:
        nc (int): number of channels in input images
        nh (int): number of hidden units in the LSTM
        nclass (int): number of classes
        height (int): height of input images
        LeakyRelu (bool): use LeakyReLU or not
    """
    def __init__(self, nc, nh, nclass, height, LeakyRelu=False):
        super(CRNN, self).__init__()

        self.cnn = self._build_cnn(nc, LeakyRelu)
        self.avg_pooling = nn.AvgPool2d(kernel_size=(height // 4, 1), stride=(height // 4, 1))
        self.rnn = nn.Sequential(
            BiLSTM(512, nh, nh),
            BiLSTM(nh, nh, nclass)
        )

    def _build_cnn(self, nc, LeakyRelu):
        cnn = nn.Sequential()
        channels = [64, 128, 256, 256, 512, 512, 512]
        kernel_size = [3] * len(channels)
        padding = [1] * len(channels)
        stride = [1] * len(channels)

        for i in range(len(channels)):
            nIn = nc if i == 0 else channels[i - 1]
            nOut = channels[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if i in {2, 4, 6}:  # Apply BatchNorm only to specific layers
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if LeakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(inplace=True))
            if i in {0, 1, 3, 5}:  # Apply MaxPooling only to specific layers
                pooling_kernel = (2, 2)
                pooling_stride = (1, 2) if i < 2 else (2, 1)
                pooling_padding = (1, 0) if i < 2 else (0, 1)
                cnn.add_module(f'pooling{i}', nn.MaxPool2d(pooling_kernel, pooling_stride, pooling_padding))

        return cnn

    """
    input: the input image with the size of [batch, channel, height, width]
    return: the probability of each class for each step
    """
    def forward(self, input):
        conv = self.cnn(input)
        conv = self.avg_pooling(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output