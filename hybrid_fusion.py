import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


class HybridFusionModel(nn.Module):
    def __init__(self):
        super(HybridFusionModel, self).__init__()

        # BERT for text
        self.bert = BertModel.from_pretrained('distilbert-base-uncased')

        # ResNet for image
        self.resnet = models.resnet34(pretrained=True)

        # Shallow hybrid fusion
        self.fc_hybrid = nn.Linear(768 + 1000, 512)

        # Late fusion after deeper processing
        self.bert_fc = nn.Linear(768, 512)
        self.resnet_fc = nn.Linear(1000, 512)
        self.fc_final = nn.Linear(512 + 512, 2)  # Binary classification

    def forward(self, input_ids, attention_mask, images):
        # Shallow fusion
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        image_features = self.resnet(images)
        combined_hybrid = torch.cat((text_features, image_features), dim=1)
        shallow_fusion = self.fc_hybrid(combined_hybrid)

        # Late fusion after deeper processing
        text_features_deep = self.bert_fc(text_features)
        image_features_deep = self.resnet_fc(image_features)
        combined_deep = torch.cat((text_features_deep, image_features_deep), dim=1)

        # Final classification
        return self.fc_final(combined_deep)
