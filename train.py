# Author : skjang@mnc.ai
# Date : 2021-11-08

from data import *
from utils import *
import logging
import pandas as pd
import os, glob, time, datetime
import copy, shutil
import argparse

import random
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
import warnings 
warnings.filterwarnings("ignore")

def run_epoch(model, dataloader, criterion, optimizer=None, epoch=0, scheduler=None, device='cpu'):
    import torchmetrics.functional as clmetrics
    from torchmetrics import Precision, Accuracy, Recall
    #import pytorch_lightning.metrics.functional.classification as clmetrics
    #from pytorch_lightning.metrics import Precision, Accuracy, Recall
    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics = Accumulator()
    cnt = 0
    total_steps = len(dataloader)
    steps = 0
    running_corrects = 0
    

    accuracy = Accuracy()
    precision = Precision(num_classes=2)
    recall = Recall(num_classes=2)

    preds_epoch = []
    labels_epoch = []
    for inputs, labels in dataloader:
        steps += 1
        inputs = inputs.to(device) # torch.Size([2, 1, 224, 224])
        labels = labels.to(device).unsqueeze(1).float() ## torch.Size([2, 1])

        outputs = model(inputs) # [batch_size, nb_classes]

        loss = criterion(outputs, labels)

        if optimizer:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        preds_epoch.extend(torch.sigmoid(outputs).tolist())
        labels_epoch.extend(labels.tolist())
        threshold = 0.5
        prob = (torch.sigmoid(outputs)>threshold).long()
        
        conf = torch.flatten(clmetrics.confusion_matrix(prob, labels, num_classes=2))
        tn, fp, fn, tp = conf

        metrics.add_dict({
            'data_count': len(inputs),
            'loss': loss.item() * len(inputs),
            'tp': tp.item(),
            'tn': tn.item(),
            'fp': fp.item(),
            'fn': fn.item(),
        })
        cnt += len(inputs)

        if scheduler:
            scheduler.step()
        del outputs, loss, inputs, labels, prob
    logger.info(f'cnt = {cnt}')

    metrics['loss'] /= cnt

    def safe_div(x,y):
        if y == 0:
            return 0
        return x / y
    _TP,_TN, _FP, _FN = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
    acc = (_TP+_TN)/cnt
    sen = safe_div(_TP , (_TP + _FN))
    spe = safe_div(_TN , (_FP + _TN))
    prec = safe_div(_TP , (_TP + _FP))
    metrics.add('accuracy', acc)
    metrics.add('sensitivity', sen)
    metrics.add('specificity', spe)
    metrics.add('precision', prec)

    auc = roc_auc_score(labels_epoch, preds_epoch)
    aupr = average_precision_score(labels_epoch, preds_epoch)
    metrics.add('auroc', auc)
    metrics.add('aupr', aupr)

    logger.info(metrics)

    return metrics, preds_epoch, labels_epoch

def train_and_eval(args, fold_num, eff_net='b0', max_epoch=10, save_path='.', batch_size=8, multigpu=True): #TODO
    os.makedirs(save_path, exist_ok=True) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'DEVICE: {device}')
    logger.info('====='*10)
    logger.info(' '*20 + f'FOLD: {fold_num} / 10')
    logger.info('====='*10)

    train_dataset, trainloader, validloader, testloader = get_dataloaders(args, ImageDataset, batch=batch_size, root='.', fold_num=fold_num, multinode=multigpu, augment=False)
    
    criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLossV2()
    
    rs = dict()

    if args.trained_model is not None :
        # perform only evaluation
        model = get_trained_model(args.trained_model)
        model.to(device)

    else: 
        # load pretrained model and retrain using our data
        model = EfficientNet.from_pretrained(f'efficientnet-{eff_net}', num_classes=1, in_channels=1)
        model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #weight_decay=1e-5
        #optimizer = optim.Adam(model.parameters(), lr=0.001)

        logger.info(f'optimizer: {optimizer}')
        logger.info(f'criterion: {criterion}')

        #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scheduler = CosineWarmupLR(optimizer=optimizer, epochs=max_epoch, iter_in_one_epoch=len(trainloader), lr_min=0.0001,
                                warmup_epochs=0)

        logger.info(f'scheduler: {scheduler}')

        if multigpu :
            model = nn.DataParallel(model)

        since = time.time()

        for epoch in range(max_epoch):
            logger.info('-' * 10)
            logger.info('Start Epoch {}/{}'.format(epoch, max_epoch - 1))

            logger.info('====='*10)
            logger.info(' '*20 + f'START EPOCH {epoch}')
            logger.info('====='*10)

            model.train()
            logger.info('-----'*10)
            logger.info(' '*20 + f'TRAINING')
            logger.info('-----'*10)
            rs['train'],_ ,_ = run_epoch(model, trainloader, criterion, optimizer=optimizer, scheduler=scheduler, device=device)

            logger.info('-----'*10)
            logger.info(' '*20 + f'VALIDATION')
            logger.info('-----'*10)

            model.eval()
            with torch.no_grad():
                rs['valid'], preds_valid, labels_valid = run_epoch(model, validloader, criterion, optimizer=None, device=device)

            logger.info(
                f'[ EPOCH {epoch} ]'
                f'[ TRAIN ] loss={rs["train"]["loss"]:.3f}, accuracy={rs["train"]["accuracy"]:.3f} '
                f'[ VALID ] loss={rs["valid"]["loss"]:.3f}, accuracy={rs["valid"]["accuracy"]:.3f}, precision={rs["valid"]["precision"]:.3f}, recall={rs["valid"]["sensitivity"]:.3f}'
            )
            # tensorboard
            writer.add_scalars(f"Accuracy/fold{fold_num}",{'train':rs["train"]["accuracy"], 'valid':rs["valid"]["accuracy"]}, epoch)
            writer.add_scalars(f"Precision/fold{fold_num}",{'train':rs["train"]["precision"],'valid':rs["valid"]["precision"]}, epoch)
            writer.add_scalars(f"Specificity/fold{fold_num}",{'train':rs["train"]["specificity"],'valid':rs["valid"]["specificity"]}, epoch)
            writer.add_scalars(f"Sensitivity/fold{fold_num}",{'train':rs["train"]["sensitivity"],'valid':rs["valid"]["sensitivity"]}, epoch)
            writer.add_scalars(f"AUPR/fold{fold_num}",{'train':rs["train"]["aupr"], 'valid':rs["valid"]["aupr"]}, epoch)
            writer.add_scalars(f"AUROC/fold{fold_num}",{'train':rs["train"]["auroc"], 'valid':rs["valid"]["auroc"]}, epoch)
            writer.add_scalars(f"Loss/fold{fold_num}",{'train':rs["train"]["loss"], 'valid':rs["valid"]["loss"]}, epoch)
            writer.flush()

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    logger.info('====='*10)
    logger.info(' '*20 + f'EVALUATION')
    logger.info('====='*10)
    model.eval()
    with torch.no_grad():
        rs['test'], preds_test, labels_test = run_epoch(model, testloader, criterion, optimizer=None, device=device)

    if args.trained_model is None :
        os.makedirs(f'{save_path}/weights/', exist_ok=True)
        torch.save({
                            'epoch': epoch,
                            'log': {
                                'train': rs['train'].get_dict(),
                                'valid': rs['valid'].get_dict(),
                                'test': rs['test'].get_dict(),
                            },
                            'optimizer': optimizer.state_dict(),
                            'model': model.state_dict(),
                        }, f'{save_path}/weights/model_weights_{eff_net}_fold{fold_num}.pth')


    logger.info(f'[ TEST ] loss={rs["test"]["loss"]:.3f}, accuracy={rs["test"]["accuracy"]:.3f}, AUROC={rs["test"]["auroc"]:.3f}, sensitivity={rs["test"]["sensitivity"]:.3f}')
    writer.add_scalars(f"Accuracy/fold{fold_num}",{f'test':rs["test"]["accuracy"] },max_epoch)
    writer.add_scalars(f"Precision/fold{fold_num}",{f'test':rs["test"]["precision"] },max_epoch)
    writer.add_scalars(f"Specificity/fold{fold_num}",{f'test':rs["test"]["specificity"] },max_epoch)
    writer.add_scalars(f"Sensitivity/fold{fold_num}",{f'test':rs["test"]["sensitivity"] },max_epoch)
    writer.add_scalars(f"AUPR/fold{fold_num}",{f'test':rs["test"]["aupr"] },max_epoch)
    writer.add_scalars(f"AUROC/fold{fold_num}",{f'test':rs["test"]["auroc"] },max_epoch)
    writer.add_scalars(f"Loss/fold{fold_num}",{f'test':rs["test"]["loss"] },max_epoch)
    writer.flush()

    # record final evaluation metric
    savedir = f'{save_path}/metrics'
    os.makedirs(savedir,exist_ok=True) 
    
    def save_metric_csv(x):
        dic = rs[x].get_dict()
        pd.DataFrame.from_dict(dic, orient='index').to_csv(os.path.join(savedir, f"metric_fold{fold_num}_")+f'{x}.csv', header=False)
    save_metric_csv('test')
    pd.DataFrame({'pred':preds_test, 'label':labels_test }).to_csv(os.path.join(savedir, f"testset_prediction_fold{fold_num}.csv"),index=False)
    return None

def get_trained_model(model_path):
    # load trained parameter
    ch = torch.load(model_path)
    state_dict = ch['model'] # dict_keys(['epoch', 'log', 'optimizer', 'model'])

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model = EfficientNet.from_pretrained(f'efficientnet-b3', num_classes=1, in_channels=1 if disease=='Sinusitis' else 3)
    model.load_state_dict(new_state_dict)

    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None,
                        help='Index of 10 fold cross-validation to train model')
    parser.add_argument('--trained_model', type=str, default=None,
                        help='Load a trained model in the path and perform only evaluation, without training process')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    disease = ['Sinusitis','Oral_cancer'][0]
    today_datever = datetime.datetime.now().strftime("%y%m%d")
    logger = get_logger(f'{disease}_EfficientNet_{today_datever}', resetlogfile=True)
    logger.setLevel(logging.INFO)

    tb = f'./log/{today_datever}/tensorboard'
    if os.path.exists(tb):
        shutil.rmtree(tb)
    os.makedirs(tb)
    writer = SummaryWriter(tb)

    if args.fold is None:
        cv_fold = range(10)
    else:
        assert isinstance(args.fold,int)
        cv_fold = [args.fold]
        
    np.random.seed(0)
    torch.manual_seed(0)

    since = time.time()

    save_validationlist()
    
    for ii in cv_fold:
        _ = train_and_eval(args, fold_num=ii, eff_net='b3', max_epoch=20, batch_size=128, multigpu=True, save_path=f'./log/{today_datever}')

    writer.close()
    time_elapsed = time.time() - since
    logger.info('Complete in {:.0f}m {:.0f}s !!'.format(time_elapsed // 60, time_elapsed % 60))
