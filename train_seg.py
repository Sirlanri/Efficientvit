import sys
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
from sklearn.model_selection import KFold

from configs.seg.train_seg_configs import *
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 参数区

# 定义用于保存loss的列表
loss_list = []

def compute_metrics(preds, labels, num_classes=12):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    iou = jaccard_score(labels_flat, preds_flat, average='macro', labels=range(num_classes))
    accuracy = accuracy_score(labels_flat, preds_flat)
    return iou, accuracy


class SegmentationTransforms:
    def __init__(self):
        # Define transformations for the source image including ColorJitter
        self.image_transforms = transforms.Compose([
            #颜色变换
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Add ColorJitter only for the image
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
        ])

        # Define transformations for the mask, excluding color-related transformations
        self.mask_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Ensure the same flips as the image
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),  # Ensure the same rotation as the image
            # Note: No ColorJitter for the mask
        ])

    def __call__(self, image, mask):
        # 注：如果发生形态的变化，请确保mask和image的变化一致
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)
        
        return image, mask

class SegTransforms2:
    def __init__(self, p_horizontal_flip=0.5, p_vertical_flip=0.5, degrees=45):
        """
        初始化方法，设置随机变换的概率和旋转角度范围。
        
        :param p_horizontal_flip: 水平翻转的概率，默认为0.5。
        :param p_vertical_flip: 垂直翻转的概率，默认为0.5。
        :param degrees: 旋转的角度范围，默认为45度。
        """
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p_horizontal_flip),
            transforms.RandomVerticalFlip(p=p_vertical_flip),
            transforms.RandomRotation(degrees=degrees)
        ])
        self.image_transforms = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)

    def __call__(self, img, mask):
        """
        对图像和掩码应用相同的随机变换，通过先拼接后分离的方式。
        """
        # 将图像和掩码转换为张量，并按通道拼接
        img_tensor = transforms.ToTensor()(img)
        mask_tensor = transforms.ToTensor()(mask)
        
        # 假设图像为RGB，掩码为单通道，拼接成新的张量
        concatenated_tensor = torch.cat((img_tensor, mask_tensor), dim=0)
        
        # 应用变换
        transformed_concatenated = self.transforms(concatenated_tensor)
        
        # 分离变换后的张量回原始的图像和掩码
        transformed_img = transformed_concatenated[:3, :, :]  # 假设原图是RGB，所以取前3个通道
        transformed_mask = transformed_concatenated[3:, :, :]  # 掩码是剩下的通道
        
        # 转换回PIL.Image格式，如果需要的话
        transformed_img = transforms.ToPILImage()(transformed_img)
        transformed_mask = transforms.ToPILImage()(transformed_mask)
        return transformed_img, transformed_mask

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

#能够交叉验证的数据集
class AutoSemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self,root_dir, image_processor, transform=None, train=True):
        """
        重写啦~ 不会手动划分数据集啦
        Args:
            root_dir (string): Root directory of the dataset containing the images + masks.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir=root_dir
        self.image_processor = image_processor
        self.transform = transform

        
        self.img_dir = os.path.join(self.root_dir, "images")
        self.mask_dir = os.path.join(self.root_dir, "masks")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read masks
        mask_file_names = []
        for root, dirs, files in os.walk(self.mask_dir):
            mask_file_names.extend(files)
        self.masks = sorted(mask_file_names)

        assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        segmentation_map = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")
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

    transform = None

    #总数据集
    rootDataset=AutoSemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, transform=transform, train=True)

    #交叉验证
    kf = KFold(n_splits=N_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(rootDataset)):
        print(f" Fold {fold + 1}/{N_splits}")
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_dataloader = DataLoader(train_sampler, batch_size=Batch_Size,num_workers=Num_workers,pin_memory=True, shuffle=True)
        valid_dataloader = DataLoader(test_sampler, batch_size=Batch_Size,num_workers=Num_workers,pin_memory=True)


        train_dataloader = DataLoader(rootDataset, sampler=train_sampler,batch_size=Batch_Size,num_workers=Num_workers,pin_memory=True)
        valid_dataloader = DataLoader(rootDataset, sampler=test_sampler,batch_size=Batch_Size,num_workers=Num_workers,pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        model = create_seg_model(Model_size, "tire12", pretrained=False)

        #确认路径存在，不存在则创建
        if not os.path.exists(out_weights_path):
            os.makedirs(out_weights_path)
        if not os.path.exists(TensorBoard_dir):
            os.makedirs(TensorBoard_dir)
        # 从文件中加载最新一轮的模型
        start_epoch=load_checkpoint(model, out_weights_path)+1

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

        #检查当前系统是否为Linux，如果是，则使用torch.compile
        if sys.platform.startswith('linux'):
            print('Current system is Linux, using torch.compile')
            #model=torch.compile(model,mode='max-autotune')
        else:
            print('Current system is not Linux, disabled torch.compile')
        model.to(device)

        # TensorBoard setup
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # log_dir = f'/path/to/logs/{current_time}'  # Customize the path as needed
        log_dir = TensorBoard_dir
        writer = SummaryWriter(log_dir)
        epoch_iou, epoch_accuracy = [], []

        for epoch in range(start_epoch,epochs):  # loop over the dataset multiple times
            model.train()
            print("Epoch:", epoch)
            epoch_loss = 0.0
            ious, accuracies = [], []

            for idx, batch in enumerate(tqdm(train_dataloader,desc="Training")):
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
                for idx, batch in enumerate(tqdm(valid_dataloader,desc="Validating")):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(pixel_values)
                    upsampled_outputs = nn.functional.interpolate(
                        outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )

                    loss = criterion(upsampled_outputs, labels)

                    val_loss += loss.item()

                    predicted = upsampled_outputs.argmax(dim=1)
                    if IS_IOU_ACC:
                        batch_iou, batch_accuracy = compute_metrics(predicted.cpu().numpy(), labels.cpu().numpy())
                        val_ious.append(batch_iou)
                        val_accuracies.append(batch_accuracy)

            # Calculate and print average validation loss and metrics
            avg_val_loss = val_loss / len(valid_dataloader)
            if IS_IOU_ACC:
                avg_val_iou = sum(val_ious) / len(val_ious)
                avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
                print(f"Validation Avg IoU: {avg_val_iou:.4f}, Avg Accuracy: {avg_val_accuracy:.4f}")
                writer.add_scalar('IoU/val', avg_val_iou, epoch)
                writer.add_scalar('Accuracy/val', avg_val_accuracy, epoch)

            print(f"Validation - Avg Loss: {avg_val_loss:.4f}")
            # Log validation metrics
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            

            # 保存模型
            save_checkpoint(model, out_weights_path, epoch, avg_val_loss, max_to_save)