import torch 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm 
import numpy as np 
import argparse
import random
import os 
import cv2
import matplotlib.pyplot as plt
from timm.loss import LabelSmoothingCrossEntropy
from dataloader import TestData, ValidData
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Hazy Density Classification")
    parser.add_argument('--data_dir', type=str, help='path to data')
    parser.add_argument('--no_cuda', action='store_true', help='not use cuda')
    parser.add_argument('--bs', dest='batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()
    return args

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

def test(model, dataloader, device, criterion, batch_size, classes,
        num_visual=(2, 2), visual=True):
    test_loss = 0.0
    class_correct = list(0 for i in range(len(classes)))
    class_total = list(0 for i in range(len(classes)))
    model.eval()
    target_list = []
    pred_list = []

    total_visual = num_visual[0] * num_visual[1]
    iter_visual = 0
    fig, axes = plt.subplots(num_visual[0], num_visual[1], 
              figsize=(num_visual[1]*5, num_visual[0]*5))

    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad(): 
            output = model(data)
            loss = criterion(output, target)
        test_loss = loss.item() * data.size(0)
        pred_ids, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        if len(target) == batch_size:
            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        target_list.append(target.cpu().detach().numpy())
        pred_list.append(pred.cpu().detach().numpy())

        if visual:
          if iter_visual < total_visual-1:
              for i in range(len(data)):
                sample = data[i].cpu().detach().numpy()
                sample = (sample - sample.min()) / (sample.max() - sample.min())
                sample = sample * 255
                sample = sample.transpose(1, 2, 0).astype(np.uint8)
                sample = cv2.resize(sample, (400, 400))
                sample = cv2.rectangle(sample, (10, 10), (200, 100), (255, 255, 255), -1)
                sample = cv2.rectangle(sample, (10, 10), (200, 100), (0,0,0), 5)
                sample = cv2.putText(sample, f'Prediction: {pred_ids.data[i]:.2f}', (20,30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                sample = cv2.putText(sample, f'Confidence: {pred.data[i]}', (20,55),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                sample = cv2.putText(sample, f'Label: {target.data[i]}', (20,80),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                

                x = iter_visual % num_visual[0]
                y = int(iter_visual / num_visual[1])
                
                ax = axes[y][x]
                ax.imshow(sample)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                iter_visual += 1
                if iter_visual == total_visual:
                  break

    if visual:
      plt.savefig('test.jpg')
      plt.show()

    test_loss = test_loss / dataloader.dataset.len

    print('Test Loss: {:.4f}'.format(test_loss))
    for i in range(len(classes)):
        if class_total[i] > 0:
            print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
                classes[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print("Test accuracy of %5s: NA" % (classes[i]))
    print("Test Accuracy of %2d%% (%2d/%2d)" % (100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
    print(classification_report(np.concatenate(target_list), np.concatenate(pred_list)))
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

        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = torch.load('res34_haze_new.pt').to(device)


    test_loader = DataLoader(TestData(args.data_dir), batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    criterion = LabelSmoothingCrossEntropy()
    criterion = criterion.cuda()    
    
    classes = get_classes('/content/drive/MyDrive/data/train')

    test(model, test_loader, device, criterion, args.batch_size, classes)

if __name__ == "__main__":
    main()