import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


class LateFusionModel(nn.Module):
    def __init__(self):
        super(LateFusionModel, self).__init__()

        # BERT for text
        self.bert = BertModel.from_pretrained('distilbert-base-uncased')
        self.bert_fc = nn.Linear(768, 512)

        # ResNet for image
        self.resnet = models.resnet34(pretrained=True)
        self.resnet_fc = nn.Linear(1000, 512)

        # Late fusion (concatenating extracted features)
        self.fc_final = nn.Linear(512 + 512, 2)  # Binary classification

    def forward(self, input_ids, attention_mask, images):
        # Extract text features
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        text_features = self.bert_fc(text_features)

        # Extract image features
        image_features = self.resnet(images)
        image_features = self.resnet_fc(image_features)

        # Late fusion (concatenation of features)
        combined = torch.cat((text_features, image_features), dim=1)

        # Final classification
        return self.fc_final(combined)
