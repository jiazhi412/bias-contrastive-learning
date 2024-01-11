import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from debias.datasets.celeba import get_celeba, get_celeba_ex
from debias.networks.resnet_lnl import LNLBiasPredictor, LNLResNet18, LNLResNet18_feat
from debias.utils.logging import set_logging
from debias.utils.training import grad_reverse
from debias.utils.utils import (AverageMeter, MultiDimAverageMeter, accuracy,
                                pretty_dict, save_model, set_seed)
import utils


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', type=str, default='makeup')
    parser.add_argument("--eval_mode", type=str, default='unbiased')
    parser.add_argument('--n_bc', default=-1, type=int, help='number of bias-conflicting samples')
    parser.add_argument('--p_bc', default=-1, type=float, help='percentage of bias-conflicting samples')
    parser.add_argument("--balance", action="store_true")

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3)

    # hyperparameters
    parser.add_argument('--weight', type=float, default=1)

    opt = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def set_model():
    model = LNLResNet18_feat().cuda()
    predictor = LNLBiasPredictor().cuda()
    return model, predictor


def train(train_loader, model, predictor, optimizer, pred_optimizer, epoch, opt):
    model.train()
    avg_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    bias_criterion = nn.CrossEntropyLoss()

    train_iter = iter(train_loader)
    for idx, (images, labels, biases, _) in enumerate(train_iter):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        target_logits, bias_feats, _ = model(images)

        bias_logits = predictor(bias_feats)

        target_loss = criterion(target_logits, labels)
        bias_entropy_loss = (F.softmax(bias_logits) * F.log_softmax(bias_logits)).sum(1).mean()

        loss = target_loss + opt.weight * bias_entropy_loss

        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, bias_feats, _ = model(images)

        bias_feats_rev = grad_reverse(bias_feats)
        bias_logits = predictor(bias_feats_rev)

        bias_loss = bias_criterion(bias_logits, biases)

        optimizer.zero_grad()
        pred_optimizer.zero_grad()
        bias_loss.backward()
        optimizer.step()
        pred_optimizer.step()

    return avg_loss.avg


def validate(val_loader, model, opt=None, is_test=False, ce=0, train_or_test='train', results=None):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))
    output_list = []
    feature_list = []
    target_list = []
    a_list = []
    with torch.no_grad():
        for idx, (images, labels, biases, _) in enumerate(val_loader):
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            output, _, feature = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))
            feature_list.append(feature)
            output_list.append(preds)
            target_list.append(labels)
            a_list.append(biases)

    feature, output, a, target = torch.cat(feature_list), torch.cat(output_list), torch.cat(a_list), torch.cat(target_list)
    output = output.unsqueeze(1)
    a = a.unsqueeze(1)
    target = target.unsqueeze(1)

    if is_test:
        if train_or_test == 'train':
            if opt.task == "blonde":
                attrs = ['Blond_Hair']
            elif opt.task == "makeup":
                attrs = ['Heavy_Makeup']
            mi_ZA = utils.estimate_MI(feature, a)
            mi_ZY = utils.estimate_MI(feature, target) if ce != 0 else mi_ZA
            mi_YpA = utils.compute_MI_withlogits(output, a)
            mi_YpY = utils.compute_MI_withlogits(output, target)
            mi_YA = utils.compute_MI_withlogits(target, a)
            results = {
                'Time': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Attr': [attrs[0]],
                'Eval Mode': [opt.eval_mode],
                'H(Y|A)': [ce],
                'I(Z,A)': [mi_ZA],
                'I(Z,Y)': [mi_ZY],
                'I(Y\',A)': [mi_YpA],
                'I(Y\',Y)': [mi_YpY],
                'I(Y,A)': [mi_YA],
                'Train Acc': [top1.avg.item()],
                }
        else:
            results['Test Acc'] = [top1.avg.item()]
            utils.append_data_to_csv(results, os.path.join('exp_results', 'LNL', opt.exp_name, 'CelebA_LNL_trials.csv'))
    return top1.avg, attrwise_acc_meter.get_mean(), results

def main():
    opt = parse_option()
    
    if opt.task == "makeup":
        # opt.epochs = 1
        opt.epochs = 26
    elif opt.task == "blonde":
        # opt.epochs = 1
        opt.epochs = 10
    else:
        raise AttributeError()

    # exp_name = f'lnl-celeba_{opt.task}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-w{opt.weight}-seed{opt.seed}'
    exp_name = f'lnl-lr{opt.lr}-bs{opt.bs}-w{opt.weight}-seed{opt.seed}'
    # opt.exp_name = exp_name

    # output_dir = f'exp_results/{exp_name}'
    output_dir = f'exp_results/LNL/{opt.exp_name}/{opt.task}/{opt.eval_mode}/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    logging.info(f'Set seed: {opt.seed}')
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = '/nas/vista-ssd01/users/jiazli/datasets/CelebA/'

    train_loader, ce = get_celeba_ex(
        root,
        batch_size=opt.bs,
        target_attr=opt.task,
        split='train',
        aug=False,
        eval_mode=opt.eval_mode,
        n_bc=opt.n_bc,
        p_bc=opt.p_bc,
        balance=opt.balance)

    val_loaders = {}
    val_loaders['valid'], _ = get_celeba_ex(
        root,
        batch_size=256,
        target_attr=opt.task,
        split='train_valid',
        aug=False,
        eval_mode=opt.eval_mode)

    val_loaders['test'], _ = get_celeba_ex(
        root,
        batch_size=256,
        target_attr=opt.task,
        split='valid',
        aug=False,
        eval_mode=opt.eval_mode)

    model, predictor = set_model()

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    pred_optimizer = torch.optim.Adam(predictor.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    pred_scheduler = torch.optim.lr_scheduler.MultiStepLR(pred_optimizer, milestones=decay_epochs, gamma=0.1)
    logging.info(f"decay_epochs: {decay_epochs}")
    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]} pred_scheduler Learning rate: {pred_scheduler.get_last_lr()[0]}')
        loss = train(train_loader, model, predictor, optimizer, pred_optimizer, epoch, opt)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss:.4f}')

        scheduler.step()
        pred_scheduler.step()

        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            accs, valid_attrwise_accs, _ = validate(val_loader, model)

            stats[f'{key}/acc'] = accs.item()
            stats[f'{key}/acc_unbiased'] = torch.mean(valid_attrwise_accs).item() * 100
            eye_tsr = torch.eye(2)
            stats[f'{key}/acc_skew'] = valid_attrwise_accs[eye_tsr > 0.0].mean().item() * 100
            stats[f'{key}/acc_align'] = valid_attrwise_accs[eye_tsr == 0.0].mean().item() * 100

        logging.info(f'[{epoch} / {opt.epochs}] {valid_attrwise_accs} {stats}')
        for tag in val_loaders.keys():
            if stats[f'{tag}/acc'] > best_accs[tag]:
            # if stats[f'{tag}/acc_unbiased'] > best_accs[tag]:
                best_accs[tag] = stats[f'{tag}/acc_unbiased']
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(**{f'best_{tag}_{k}': v for k, v in stats.items()})

                save_file = save_path / 'checkpoints' / f'best_{tag}.pth'
                save_model(model, optimizer, opt, epoch, save_file)
            logging.info(f'[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}')

    _, _, results = validate(train_loader, model, opt=opt, is_test=True, ce=ce, train_or_test='train')
    validate(val_loaders['test'], model, opt=opt, is_test=True, ce=ce, train_or_test='test', results=results)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

    save_file = save_path / 'checkpoints' / f'last.pth'
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
