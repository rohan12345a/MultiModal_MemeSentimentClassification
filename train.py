import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

# Import fusion models
from early_fusion import EarlyFusionModel
from late_fusion import LateFusionModel
from hybrid_fusion import HybridFusionModel

# Image preprocessing
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
def load_data(df, tokenizer, max_len, image_dir):
    input_ids, attention_masks, images, labels = [], [], [], []

    for idx, row in df.iterrows():
        text = row['sentence']
        image_name = row['image_name']
        label = 1 if row['label'] == 'offensive' else 0
        
        # Tokenize the text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoding['input_ids'].squeeze(0))
        attention_masks.append(encoding['attention_mask'].squeeze(0))
        
        # Process the image
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)
        
        # Store the label
        labels.append(label)
    
    return torch.stack(input_ids), torch.stack(attention_masks), torch.stack(images), torch.tensor(labels)

# Dataset class using just torch Tensors for DataLoader
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, images, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'image': self.images[idx],
            'label': self.labels[idx]
        }

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    correct_predictions = 0
    losses = []

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    correct_predictions = 0
    losses = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            _, preds = torch.max(outputs, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss_fn(outputs, labels).item())

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    return accuracy, f1, sum(losses) / len(losses)

def run_experiment(fusion_type, train_loader, test_loader, device):
    if fusion_type == 'early':
        model = EarlyFusionModel().to(device)
    elif fusion_type == 'late':
        model = LateFusionModel().to(device)
    elif fusion_type == 'hybrid':
        model = HybridFusionModel().to(device)
    else:
        raise ValueError("Invalid fusion type. Choose from 'early', 'late', or 'hybrid'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Training
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Train loss {train_loss}, accuracy {train_acc}")

    # Evaluation
    test_acc, test_f1, test_loss = eval_model(model, test_loader, loss_fn, device)
    print(f"Test accuracy for {fusion_type} fusion: {test_acc}, F1-score: {test_f1}")

    # Save the model after training and evaluation
    model_path = f"model_{fusion_type}_fusion.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the training and testing datasets from separate files
    train_df = pd.read_csv('C:\\Users\\saksh\\Desktop\\PythonCodes\\MMAL_Project\\MMAL_Dataset\\Split Dataset-20240913T115646Z-001\\Split Dataset\\Training_meme_dataset.csv')
    test_df = pd.read_csv('C:\\Users\\saksh\\Desktop\\PythonCodes\\MMAL_Project\\MMAL_Dataset\\Split Dataset-20240913T115646Z-001\\Split Dataset\\Testing_meme_dataset.csv')
    
    # Image directory and BERT tokenizer
    image_dir = 'C:\\Users\\saksh\\Desktop\\PythonCodes\\MMAL_Project\\MMAL_Dataset\\Labelled Images-20240913T121344Z-001\\Labelled Images'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load training data
    print("Loading training data...")
    train_input_ids, train_attention_masks, train_images, train_labels = load_data(train_df, tokenizer, max_len=128, image_dir=image_dir)

    # Load testing data
    print("Loading testing data...")
    test_input_ids, test_attention_masks, test_images, test_labels = load_data(test_df, tokenizer, max_len=128, image_dir=image_dir)

    # Create DataLoader for training and testing datasets
    train_loader = DataLoader(TensorDataset(train_input_ids, train_attention_masks, train_images, train_labels), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_input_ids, test_attention_masks, test_images, test_labels), batch_size=16, shuffle=False)

    # Run experiments for each fusion type
    for fusion in ['early', 'late', 'hybrid']:
        print(f"\nRunning {fusion} fusion model:")
        run_experiment(fusion, train_loader, test_loader, device)
main.py
Displaying main.py.
