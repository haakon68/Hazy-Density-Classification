import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
from model import resnet34
from dataloader import TrainData, ValidData
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm 



def parse_args():
    parser = argparse.ArgumentParser(description="super haze detect option")
    parser.add_argument('--data_dir', type=str, help='path to data')
    parser.add_argument('--no_cuda', action='store_true', help='not use cuda')
    parser.add_argument('--bs', dest='batch_size', type=int, default=128, help='batch size')

    args = parser.parse_args()
    return args

def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        for phase in ['train', 'val']: 
            if phase == 'train':
                model.train() 
            else:
                model.eval() 
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): 
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1) 
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc =  running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) 
        print()
    time_elapsed = time.time() - since 
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

def main():
    args = parse_args()
    print("Call arguments:")
    print(args)


    torch.backends.cudnn.benchmark = True
    random.seed(6801)
    torch.manual_seed(6801)
    torch.cuda.manual_seed_all(6801)
    
    if not args.no_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resnet34().to(device)
    
    train_loader = DataLoader(TrainData(args.data_dir), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ValidData(args.data_dir), batch_size=args.batch_size, shuffle=False, num_workers=4)
    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_loader.dataset.len, "val": val_loader.dataset.len}

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    exp_lr_scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.98)

    criterion = LabelSmoothingCrossEntropy()
    criterion = criterion.cuda()

    model = train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, exp_lr_scheduler, num_epochs=100)

    example = torch.rand(1, 3, 400, 400)
    traced_script_module = torch.jit.trace(model.cpu(), example)
    traced_script_module.save("res34_haze_new.pt")

if __name__ == "__main__":
    main()