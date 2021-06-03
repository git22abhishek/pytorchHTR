import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock

# from torchsummary import summary
import numpy as np


def downsample(chan_in, chan_out, stride, pad=0):

    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, kernel_size=1, stride=stride, bias=False,
                  padding=pad),
        nn.BatchNorm2d(chan_out)
    )


class CRNNModel(nn.Module):

    def __init__(self, vocab_size, time_steps, zero_init_residual=False):
        super(CRNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.time_steps = time_steps

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2,
                               bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for i in range(0, 2)])
        self.layer2 = nn.Sequential(*[BasicBlock(64, 128, stride=2,
                                      downsample=downsample(64, 128, 2))
                                      if i == 0 else BasicBlock(128, 128)
                                      for i in range(0, 2)])
        self.layer3 = nn.Sequential(*[BasicBlock(128, 256, stride=(1, 2),
                                      downsample=downsample(128, 256, (1, 2)))
                                      if i == 0 else BasicBlock(256, 256)
                                      for i in range(0, 2)])
        self.layer4 = nn.Sequential(*[BasicBlock(256, 512, stride=(1, 2),
                                      downsample=downsample(256, 512, (1, 2)))
                                      if i == 0 else BasicBlock(512, 512)
                                      for i in range(0, 2)])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_steps, 2))

        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2,
                            bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, vocab_size + 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init_constant_(m.bn2.weight, 0)

    def forward(self, xb):

        out = self.maxpool(self.bn1(self.relu(self.conv1(xb))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        # print("CNN output before transpose:", out.shape)
        # out = out.squeeze(dim=3).transpose(1, 2)
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(out.size(0,), out.size(1), -1)
        # print("CNN output after transpose:", out.shape)

        out, _ = self.lstm(out)
        # print("LSTM output:", out.shape)
        out = self.fc1(out)
        # print("FC1 output:", out.shape)
        out = self.fc2(out)
        # print("FC2 output:", out.shape)

        return out


if __name__ == '__main__':
    from dataset import Encoder, IAM
    from dataloader import CTCDataLoader
    dataset = IAM('/mnt/d/Machine-Learning/Datasets/iamdataset/uncompressed',
                  csv_file_path='iam_df.csv')
    encoder = Encoder(dataset='IAM')

    data_loader = CTCDataLoader(dataset, encoder)
    train_loader, val_loader, test_loader = data_loader(
        split=(0.7, 0.2, 0.2), batch_size=(1, 8, 8))
    model = CRNNModel(vocab_size=79, time_steps=100)
    device = torch.device('cpu')
    checkpoint = torch.load(
        'checkpoints/training_state.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    for batch in test_loader:
        model.eval()
        images, targets, target_lengths, targets_original = batch
        preds = model(images)
        text = encoder.best_path_decode(preds, return_text=True)
        print(text, targets_original)
        break
        # model.to('cpu')
    # pytorchresnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=False)
    # summary(model=model, input_size=(1, 1024, 128), batch_size=12)
    # model.eval()
    # inp = torch.rand(1, 1, 1024, 128)
    # preds = model(inp)
    # encoder = Encoder()
    # text = encoder.decode(preds)
    # print(text)
    # outs = model.best_path_decode(inp)
    # print(outs)
    # ''.join([self.decode_map.get(letter) for letter in outs[0]])
    # nn.CTCLoss(blank=0, reduction='max', zero_infinity=True)
    # torch.nn.functional.log_softmax
    # print(model.modules)
