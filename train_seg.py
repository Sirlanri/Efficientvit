from torch.utils.data import Dataset
import os
from PIL import Image
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from efficientvit.seg_model_zoo import create_seg_model
from sklearn.metrics import jaccard_score, accuracy_score
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 参数区

root_dir = r'E:\项目数据\轮胎\分割数据集\总数据集\分割格式'

# 设置保存目录
out_weights_path = r'assets\checkpoints\tire6'
# 设置保存模型数量
max_to_save = 200  # 只保存最后n轮训练的模型
# 设置训练轮数
epochs = 300
# 定义用于保存loss的列表
loss_list = []
# 学习率
LR=0.06
#Batch_Size
Batch_Size=8

def compute_metrics(preds, labels, num_classes=6):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    iou = jaccard_score(labels_flat, preds_flat, average='macro', labels=range(num_classes))
    accuracy = accuracy_score(labels_flat, preds_flat)
    return iou, accuracy


class SegmentationTransforms:
    def __init__(self):
        # Define transformations here
        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            # Add more transformations as needed
        ])

    def __call__(self, image, mask):
        # Apply to image
        image = self.transforms(image)

        # Apply the same transformations to the mask if necessary
        # For geometric transforms, ensure identical transformations to both image and mask

        return image, mask


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, transform=None, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train
        self.transform = transform

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx])).convert("L")
        if self.transform:
            image, segmentation_map = self.transform(image, segmentation_map)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs

def load_checkpoint(model, checkpoint_dir) -> int:
    """
    从文件中加载最新一轮的模型，并维护保存列表
    """
    # 获取所有pth文件路径
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

    # 按epoch排序，并获取最新模型路径
    if checkpoint_files:
        checkpoint_files = sorted(checkpoint_files, key=lambda f: float(f.split('_')[2]))
        latest_checkpoint_path = checkpoint_files[-1]
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        #解析epoch
        epoch = int(latest_checkpoint_path.split('_')[2])
        # 加载模型
        model.load_state_dict(torch.load(latest_checkpoint_path))
        return epoch

    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0

import re

def get_epoch(filename):
    match = re.search(r"model_epoch_(\d+)_", filename)
    if match:
        return int(match.group(1))
    else:
        return None

def delete_min_epoch_file(directory):
    files = os.listdir(directory)
    epochs = [get_epoch(filename) for filename in files if filename.endswith(".pth")]

    if epochs:
        min_epoch = min(epochs)
        for filename in files:
            if get_epoch(filename) == min_epoch:
                os.remove(os.path.join(directory, filename))
                print(f"Deleted file: {filename}")
                break
    else:
        print("No files found in the directory.")


def save_checkpoint(model, checkpoint_dir, epoch, avg_val_loss, max_to_save):
    """
    保存模型并维护保存列表
    """
    # 创建模型保存路径
    model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}_loss_{avg_val_loss:.4f}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 更新loss列表
    loss_list.append(avg_val_loss)

    # 删除旧的模型，只保留最新的max_to_save个模型
    if len(loss_list) > max_to_save:
        #删除epoch最小的模型
        delete_min_epoch_file(checkpoint_dir)



if __name__ == "__main__":

    ''' dataset format
    'dataset' folder:
        - 'annotations' :
                - 'validation' subfolder
                - 'training' subfolder
        - 'images' :
                - 'validation' subfolder
                - 'training' subfolder
    
    images are simple images nothing special :D
    annotations are images, for each pixel is the class id segment. so if we have just 3 segemntation classes, the values in the image should be just 0 1 and 2  
    '''

    image_processor = SegformerImageProcessor(reduce_labels=False, size=(1024, 1024))

    transform = SegmentationTransforms()

    train_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, transform=transform,
                                                train=True)
    valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, transform=None,
                                                train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Batch_Size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = create_seg_model("b0", "tire6", pretrained=False)

    # 从文件中加载最新一轮的模型
    start_epoch=load_checkpoint(model, out_weights_path)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    model=torch.compile(model,mode='max-autotune')
    model.to(device)

    # TensorBoard setup
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # log_dir = f'/path/to/logs/{current_time}'  # Customize the path as needed
    log_dir = 'tensorboard-output-log-path'
    writer = SummaryWriter(log_dir)
    epoch_iou, epoch_accuracy = [], []

    for epoch in range(start_epoch,epochs):  # loop over the dataset multiple times
        model.train()
        print("Epoch:", epoch)
        epoch_loss = 0.0
        ious, accuracies = [], []

        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values)

            upsampled_outputs = nn.functional.interpolate(
                outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )

            loss = criterion(upsampled_outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        # Log and print epoch metrics
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)


        # Update learning rate
        scheduler.step(avg_epoch_loss)

        print(
            f"Epoch {epoch} finished: Avg Loss = {avg_epoch_loss:.4f}")

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_ious, val_accuracies = [], []
        with torch.no_grad():
            for batch in valid_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values)
                upsampled_outputs = nn.functional.interpolate(
                    outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )

                loss = criterion(upsampled_outputs, labels)

                val_loss += loss.item()

                predicted = upsampled_outputs.argmax(dim=1)

                batch_iou, batch_accuracy = compute_metrics(predicted.cpu().numpy(), labels.cpu().numpy())
                val_ious.append(batch_iou)
                val_accuracies.append(batch_accuracy)

        # Calculate and print average validation loss and metrics
        avg_val_loss = val_loss / len(valid_dataloader)
        avg_val_iou = sum(val_ious) / len(val_ious)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
        print(
            f"Validation - Avg Loss: {avg_val_loss:.4f}, Avg IoU: {avg_val_iou:.4f}, Avg Accuracy: {avg_val_accuracy:.4f}")

        # Log validation metrics
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('IoU/val', avg_val_iou, epoch)
        writer.add_scalar('Accuracy/val', avg_val_accuracy, epoch)

        # 保存模型
        save_checkpoint(model, out_weights_path, epoch, avg_val_loss, max_to_save)