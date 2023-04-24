import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from debias.datasets.celeba import get_celeba, get_celeba_ex
from debias.losses.bias_contrastive import BiasContrastiveLoss
from debias.networks.resnet import FCResNet18_feat
from debias.utils.logging import set_logging
from debias.utils.utils import (AverageMeter, MultiDimAverageMeter, accuracy,
                                save_model, set_seed, pretty_dict)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', type=str, default='makeup')
    parser.add_argument("--eval_mode", type=str, default='unbiased')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    # parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--cbs', type=int, default=64, help='batch_size of dataloader for contrastive loss')
    parser.add_argument('--lr', type=float, default=2e-5)

    # hyperparameters
    parser.add_argument('--weight', type=float, default=0.01)
    parser.add_argument('--ratio', type=int, default=10)
    # parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--bb', type=int, default=0)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def set_model(train_loader, opt):
    model = FCResNet18_feat().cuda()
    criterion = BiasContrastiveLoss(
        confusion_matrix=train_loader.dataset.confusion_matrix,
        bb=opt.bb)

    return model, criterion


def train(train_loader, cont_train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    avg_ce_loss = AverageMeter()
    avg_con_loss = AverageMeter()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    cont_train_iter = iter(cont_train_loader)
    for idx, (images, labels, biases, _) in enumerate(train_iter):
        try:
            cont_images, cont_labels, cont_biases, _ = next(cont_train_iter)
        except:
            cont_train_iter = iter(cont_train_loader)
            cont_images, cont_labels, cont_biases, _ = next(cont_train_iter)

        bsz = labels.shape[0]
        cont_bsz = cont_labels.shape[0]

        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        # print(images.size())
        logits, _, _ = model(images)
        preds = logits.data.max(1, keepdim=True)[1].squeeze(1)

        # print(logits)
        # print(preds)
        # print(labels)
        # print(biases)
        # print('kljerl')

        total_images = torch.cat([cont_images[0], cont_images[1]], dim=0)
        total_images, cont_labels, cont_biases = total_images.cuda(), cont_labels.cuda(), cont_biases.cuda()
        # print(total_images.size())
        _, cont_features, _ = model(total_images)

        f1, f2 = torch.split(cont_features, [cont_bsz, cont_bsz], dim=0)
        cont_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        ce_loss, con_loss = criterion(logits, labels, biases, cont_features, cont_labels, cont_biases)

        loss = ce_loss * opt.weight + con_loss

        avg_ce_loss.update(ce_loss.item(), bsz)
        avg_con_loss.update(con_loss.item(), bsz)
        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_ce_loss.avg, avg_con_loss.avg, avg_loss.avg


def validate(val_loader, model, opt=None, is_test=False):
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

            # print(output)
            # print(preds)
            # print(labels)
            # print(biases)
            # print('dajldjas')

            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))
            feature_list.append(feature)
            output_list.append(preds)
            target_list.append(labels)
            a_list.append(biases)
    
    import utils
    feature, output, a, target = torch.cat(feature_list), torch.cat(output_list), torch.cat(a_list), torch.cat(target_list)
    # print(feature.size())
    # print(output.size())
    # print(a.size())
    # print(target.size())
    output = output.unsqueeze(1)
    a = a.unsqueeze(1)
    target = target.unsqueeze(1)
    if is_test:
        if opt.task == "blonde":
            attrs = ['Blond_Hair']
        elif opt.task == "makeup":
            attrs = ['Heavy_Makeup']
        mi_RA = utils.estimate_MI(feature, a)
        mi_YpA = utils.compute_MI_withlogits(output, a)
        mi_YpY = utils.compute_MI_withlogits(output, target)
        mi_YA = utils.compute_MI_withlogits(output, a)
        # print(mi_RA)
        # print(mi_YpA)
        # print(mi_YpY)
        # print(mi_YA)
        data = {
            'Time': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Attr': [attrs[0]],
            'Eval Mode': [opt.eval_mode],
            'Test Acc': [top1.avg.item()],
            'I(R,A)': [mi_RA],
            'I(Y\',A)': [mi_YpA],
            'I(Y\',Y)': [mi_YpY],
            'I(Y,A)': [mi_YA],
            }
        utils.mkdir(f'exp_results/{opt.exp_name}')
        utils.append_data_to_csv(data, os.path.join('exp_results', opt.exp_name, 'CelebA_BCL_trials.csv'))
    return top1.avg, attrwise_acc_meter.get_mean()


def main():
    opt = parse_option()
    
    if opt.task == "makeup":
        opt.epochs = 40
        opt.ratio = 50
        # opt.ratio = 0
    elif opt.task == "blonde":
        opt.epochs = 10
        opt.ratio = 30
        # opt.ratio = 0
    else:
        raise AttributeError()

    exp_name = f'bc-bb{opt.bb}-celeba_{opt.task}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-cbs{opt.cbs}-w{opt.weight}-ratio{opt.ratio}-aug{opt.aug}-seed{opt.seed}-{opt.task}-{opt.eval_mode}'
    # opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    logging.info(f'Set seed: {opt.seed}')
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    # root = './data/celeba'
    # root = '/nas/vista-ssd01/users/jiazli/datasets/CelebA/raw_data/img_align_celeba'
    root = '/nas/vista-ssd01/users/jiazli/datasets/CelebA/'

    train_loader = get_celeba_ex(
        root,
        batch_size=opt.bs,
        target_attr=opt.task,
        split='train',
        aug=False,
        eval_mode=opt.eval_mode)

    cont_train_loader = get_celeba_ex(
        root,
        batch_size=opt.cbs,
        target_attr=opt.task,
        split='train',
        aug=opt.aug,
        two_crop=True,
        ratio=opt.ratio,
        given_y=True,
        eval_mode=opt.eval_mode)

    val_loaders = {}
    val_loaders['valid'] = get_celeba_ex(
        root,
        batch_size=256,
        target_attr=opt.task,
        split='train_valid',
        aug=False,
        eval_mode=opt.eval_mode)

    val_loaders['test'] = get_celeba_ex(
        root,
        batch_size=256,
        target_attr=opt.task,
        split='valid',
        aug=False,
        eval_mode=opt.eval_mode)

    model, criterion = set_model(train_loader, opt)

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]} weight: {opt.weight}')
        ce_loss, con_loss, loss = train(train_loader, cont_train_loader, model, criterion, optimizer, epoch, opt)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss} CE Loss: {ce_loss} Con Loss: {con_loss}')

        scheduler.step()

        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            accs, valid_attrwise_accs = validate(val_loader, model)

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
            print(best_accs)
            logging.info(f'[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}')

    test_loader = val_loaders['test']
    accs, valid_attrwise_accs = validate(test_loader, model, opt=opt, is_test=True)
    # print(accs)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

    save_file = save_path / 'checkpoints' / f'last.pth'
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
