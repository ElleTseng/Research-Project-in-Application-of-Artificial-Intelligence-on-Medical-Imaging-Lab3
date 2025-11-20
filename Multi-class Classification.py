import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
import warnings
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR 

warnings.filterwarnings("ignore")

TRAIN_DIR = 'train_images'
VAL_DIR = 'val_images'
TEST_DIR = 'test_images'

CLASS_LABELS = ['normal', 'bacteria', 'virus', 'COVID-19']
NUM_CLASSES = len(CLASS_LABELS)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#'resnet18'/'resnet50'/'densenet121'/'efficientnet_b3'
MODEL_NAME = 'densenet121' 

# 超參數
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 
NUM_EPOCHS = 30
MODEL_PATH = f'best_{MODEL_NAME}_final_submission_model.pth' 
EFFNET_RESOLUTION = 300 #for efficientNet

class MergedCXRDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['new_filename']
        folder = row['folder'] if 'folder' in row else TEST_DIR
        img_path = os.path.join(os.getcwd(), folder, filename)

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, filename
        else:
            label_one_hot = row[CLASS_LABELS].values.astype(float)
            label = np.argmax(label_one_hot)
            return image, torch.tensor(label, dtype=torch.long)
            
def get_data_loaders_for_final_training():
    train_df_orig = pd.read_csv('train_data.csv')
    val_df_orig = pd.read_csv('val_data.csv')
    test_df = pd.read_csv('test_data_sample.csv')

    train_df_orig['folder'] = TRAIN_DIR
    val_df_orig['folder'] = VAL_DIR
    test_df['folder'] = TEST_DIR
    
    combined_train_df = pd.concat([train_df_orig, val_df_orig], ignore_index=True)
    
    class_counts = combined_train_df[CLASS_LABELS].sum().values
    total_samples = class_counts.sum()
    print(f"最終訓練集 (Train+Val) 類別計數: {dict(zip(CLASS_LABELS, class_counts))}")

    weights = total_samples / (NUM_CLASSES * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"計算出的類別權重 (基於 Train+Val): {class_weights.cpu().numpy()}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        #transforms.RandomResizedCrop(EFFNET_RESOLUTION, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        #transforms.Resize(EFFNET_RESOLUTION + 32),
        transforms.CenterCrop(224),
        #transforms.CenterCrop(EFFNET_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MergedCXRDataset(combined_train_df, transform=train_transform, is_test=False)
    test_dataset = MergedCXRDataset(test_df, transform=val_test_transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader, class_weights, test_df

def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features

        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.features.denseblock3.parameters():
             param.requires_grad = True
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
            
        for param in model.features[8].parameters(): 
             param.requires_grad = True
        for param in model.features[11].parameters(): 
             param.requires_grad = True

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35), 
            nn.Linear(num_ftrs, NUM_CLASSES)
        )

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        
        #for param in model.parameters():
            #param.requires_grad = False
            
        #for param in model.features[7].parameters():
            #param.requires_grad = True
            
        #for param in model.features[8].parameters():
            #param.requires_grad = True

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), 
            nn.Linear(num_ftrs, NUM_CLASSES)
        )

    else:
        raise ValueError(f"error")

    model.to(DEVICE)
    return model

def final_train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    start_time = time.time()
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"開始最終訓練 {MODEL_NAME} (在 Train+Val 數據上，共 {num_epochs} Epochs)...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train/Final)")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = running_loss / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Macro-F1: {train_macro_f1:.4f}, Train Acc: {train_acc:.4f} (LR: {current_lr:.2e})")

    torch.save(model.state_dict(), MODEL_PATH)
    
    end_time = time.time()
    print(f"訓練完成. 最終模型已儲存至 {MODEL_PATH}. 總耗時: {end_time - start_time:.2f} 秒.")

def make_predictions(model, test_loader, test_df):
    model.load_state_dict(torch.load(MODEL_PATH))
        
    model.eval()
    
    probability_predictions = {}
    hard_predictions = {}       

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Making predictions"):
            inputs = inputs.to(DEVICE)
            
            outputs = model(inputs)
            probabilities = nn.Softmax(dim=1)(outputs).cpu().numpy()

            for filename, probs in zip(filenames, probabilities):
                probability_predictions[filename] = probs
                
                pred_index = np.argmax(probs)
                pred_one_hot = np.zeros(NUM_CLASSES, dtype=int)
                pred_one_hot[pred_index] = 1
                hard_predictions[filename] = pred_one_hot

    #Softmax
    submission_prob_df = test_df[['new_filename']].copy()

    for index, row in submission_prob_df.iterrows():
        filename = row['new_filename']
        if filename in probability_predictions:
            probs = probability_predictions[filename]
            submission_prob_df.loc[index, CLASS_LABELS] = probs
            
    output_prob_filename = f'{MODEL_NAME}_final_probabilities_7.csv'
    submission_prob_df.to_csv(output_prob_filename, index=False)
    print(f"\n--- 預測完成. 結果已儲存到 {output_prob_filename} (Softmax 概率) ---")
    print("Softmax 概率文件範例 (前 5 行):")
    print(submission_prob_df.head())

    #Hard Prediction
    submission_hard_df = test_df[['new_filename']].copy()
    for col in CLASS_LABELS:
        submission_hard_df[col] = 0

    for index, row in submission_hard_df.iterrows():
        filename = row['new_filename']
        if filename in hard_predictions:
            pred_array = hard_predictions[filename]
            submission_hard_df.loc[index, CLASS_LABELS] = pred_array

    output_hard_filename = f'{MODEL_NAME}_final_hard_predictions_7.csv'
    submission_hard_df.to_csv(output_hard_filename, index=False)
    print(f"\n--- 預測完成. 結果已儲存到 {output_hard_filename} (硬預測 1/0) ---")
    print("硬預測文件範例 (前 5 行):")
    print(submission_hard_df.head())

if __name__ == '__main__':
    print("======================================================")
    print("--- 實施最終提交模型訓練策略 (Train + Val 合併) ---")
    print(f"模型: {MODEL_NAME}")
    print(f"訓練數據: Train + Val 合併 ({NUM_EPOCHS} Epochs)")
    print(f"優化點: 強化擴增, 解凍 Layer3/4, CosineLR Scheduler")
    print(f"使用設備: {DEVICE}")
    print("======================================================")

    print("\n--- 1. 數據載入與準備 (Train + Val 合併) ---")
    train_loader, test_loader, class_weights, test_df = get_data_loaders_for_final_training()

    print("\n--- 2. 模型初始化與訓練設定 ---")
    model = get_model(MODEL_NAME) 
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可訓練參數總數 (Layer 3 + 4 + FC): {trainable_params / 1e6:.2f} 百萬")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    print("\n--- 3. 最終訓練開始 ---")
    final_train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    print("\n--- 4. 進行測試預測 ---")
    make_predictions(model, test_loader, test_df)