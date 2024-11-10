import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


class EarlyFusionModel(nn.Module):
    def __init__(self):
        super(EarlyFusionModel, self).__init__()

        # BERT for text
        self.bert = BertModel.from_pretrained('distilbert-base-uncased')

        # ResNet for image
        self.resnet = models.resnet34(pretrained=True)

        # Early fusion (concatenating raw features)
        self.fc_early = nn.Linear(768 + 1000, 512)
        self.fc_final = nn.Linear(512, 2)  # Binary classification

    def forward(self, input_ids, attention_mask, images):
        # Extract text features
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]

        # Extract image features
        image_features = self.resnet(images)

        # Early fusion (concatenation of text and image features)
        combined = torch.cat((text_features, image_features), dim=1)
        combined = self.fc_early(combined)

        # Final classification
        return self.fc_final(combined)
