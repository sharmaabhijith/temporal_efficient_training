import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from models.VGG_models import *
from models.resnet_models import *
from torch.utils.data import Subset
from functions import TET_loss, seed_all, get_logger
from Preprocess import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=150,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T',
                    '--time',
                    default=4,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--dataset',
                    default="cifar100",
                    type=str,
                    metavar='N',
                    help='Name of the dataset')
parser.add_argument('--reference_models',
                    default=4,
                    type=int,
                    metavar='N',
                    help='Number of reference models')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lamb',
                    default=1e-3,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, args):
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for (images, labels) in train_loader:
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        mean_out = outputs.mean(1)
        if args.TET:
            loss = TET_loss(outputs,labels,criterion,args.means,args.lamb)
        else:
            loss = criterion(mean_out,labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc

if __name__ == '__main__':
    seed_all(args.seed)
    primary_model_path = os.path.join("saved_models", args.dataset, "resnet18", f"ref_models_{args.reference_models}")
    for model_idx in range(0, args.reference_models+1):
        full_model_path = os.path.join(primary_model_path, f"model_{model_idx}")
        if os.path.exists(full_model_path) is False:
            print("Creating model directory:", full_model_path)
            os.makedirs(full_model_path)
        save_path = os.path.join(full_model_path, "ReNetSNN.pth")
        if model_idx==0:
        # Load the dataset using the specified parameters
            dataset = load_dataset(args.dataset)
            try:
                data_split_file = os.path.join(primary_model_path, "data_splits.pkl")
                with open(data_split_file, 'rb') as file:
                    data_split_info = pickle.load(file)
                logger.info("Data split information successfully loaded:")
            except FileNotFoundError:
                logger.info(f"Error: The file '{data_split_file}' does not exist")
        # Creating dataloader
        train_idxs = data_split_info[model_idx]["train"]
        test_idxs = data_split_info[model_idx]["test"]
        logger.info(
            f"Training model {model_idx}: Train size {len(train_idxs)}, Test size {len(test_idxs)}"
        )
        logger.info("Creating dataloader...")
        train_loader = get_dataloader_from_dataset(args.dataset, Subset(dataset, train_idxs), batch_size=args.batch_size, train=True)
        test_loader = get_dataloader_from_dataset(args.dataset, Subset(dataset, test_idxs), batch_size=args.batch_size, train=False)

        model = resnet19(num_classes=100)

        parallel_model = torch.nn.DataParallel(model)
        parallel_model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

        best_acc = 0
        best_epoch = 0
        
        logger = get_logger('exp.log')
        logger.info('start training!')
        
        for epoch in range(args.epochs):
        
            loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, args)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc ))
            scheduler.step()
            facc = test(parallel_model, test_loader, device)
            logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch , args.epochs, facc ))

            if best_acc < facc:
                best_acc = facc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_path)
            logger.info('Best Test acc={:.3f}'.format(best_acc ))
            print('\n')
