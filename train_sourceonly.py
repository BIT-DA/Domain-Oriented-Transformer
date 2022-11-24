import argparse
import os
import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


import network, loss
from data_list import ImageList, ImageList_idx

from sklearn.metrics import confusion_matrix
from timm_diy.models import create_model
from timm_diy.data import create_transform

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75, weight_decay=1e-3):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = weight_decay
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize((size,size), interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()

    dsets["source"] = ImageList_idx(txt_src, transform=build_transform(True, args))
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["test"] = []
    dset_loaders["test"] = []
    for i in args.test_dset_path:
        txt_test = open(i).readlines()
        dsets["test"].append(ImageList(txt_test, transform=build_transform(False, args)))
        dset_loaders["test"].append(DataLoader(dsets["test"][-1], batch_size=train_bs * 2, shuffle=False,
                                               num_workers=args.worker,
                                               drop_last=False))
    return dset_loaders

def cal_acc(loader, model, visda=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if inputs.size(0) % 2 == 1:
                inputs_a = torch.zeros(1, 3, 224, 224).cuda()
                inputs = torch.cat((inputs, inputs_a), dim=0)
                outputs = model(inputs)
                outputs = outputs[:-1]
            else:
                outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if visda:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100

def train(args):
    dset_loaders = data_load(args)

    if args.model == 'vit_small':
        model = create_model("vit_small_patch16_224", pretrained=False, num_classes=args.class_num
        )
        pretrained_model = './pretrained/deit_small_distilled_patch16_224.pth' # we adopt the distilled version for better performance
    elif args.model == 'vit_base':
        model = create_model("vit_base_patch16_224", pretrained=False, num_classes=args.class_num
        )
        pretrained_model = './pretrained/deit_base_distilled_patch16_224.pth'
    pretrained = torch.load(pretrained_model)
    del pretrained['head.weight'], pretrained['head.bias']
    del pretrained['head_dist.weight'], pretrained['head_dist.bias'] # since pretrained model has an additional head
    del pretrained['dist_token']
    pos_embed = pretrained['pos_embed'].data
    pos_embed = torch.cat([pos_embed[:,0:1],pos_embed[:,2:]],dim=1)        
    pretrained['pos_embed'] = pos_embed
    model.load_state_dict(pretrained, strict=False)

    # print(torch.cuda.is_available())
    model = model.cuda()
    
    learning_rate = args.lr
    param_group = []
    if args.tentimes:
        for k, v in model.named_parameters():
            if k.find('head') != -1:
                param_group += [{'params': v, 'lr': learning_rate*10}]
            else:
                param_group += [{'params': v, 'lr': learning_rate}]
    else:
        for k, v in model.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    criterion = nn.CrossEntropyLoss()

    interval_iter = 2000
    max_iter = args.max_epoch * interval_iter
    iter_num = 0
    mixup_fn = None
    if args.mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.class_num)
        criterion = SoftTargetCrossEntropy()

    model.train()
    sum_cls_loss = 0.0
    while iter_num < max_iter:
        try:
            inputs_source, labels_source, _ = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source, _ = iter_source.next()

        if mixup_fn is not None:
            inputs_source, labels_source = mixup_fn(inputs_source.cuda(), labels_source.cuda())
        else:
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, weight_decay=args.weight_decay)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = model(inputs_source) 
        cls_loss = criterion(outputs_source, labels_source)

        total_loss = cls_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        sum_cls_loss += cls_loss.item()
        #print(iter_num)
        if iter_num % 100 == 0:
            log_str = 'Iter: {}, ClsLoss = {:.3f}'.format(iter_num, sum_cls_loss/100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            sum_cls_loss = 0.

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            for k in range(len(args.test_dset_path)):
                if args.dset == 'visda2017':
                    acc_s_te, acc_list = cal_acc(dset_loaders['test'][k], model, True)
                    log_str = 'Task: {}({}), Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, args.name_test[k], iter_num,
                                                                                max_iter, acc_s_te) + '\n' + acc_list
                else:
                    acc_s_te = cal_acc(dset_loaders['test'][k], model, False)
                    log_str = 'Task: {}({}), Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, args.name_test[k], iter_num,
                                                                                max_iter, acc_s_te)

                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                print(log_str + '\n')

            model.train()
    
    if args.save:
        if args.model == 'vit_small':
            torch.save(model.state_dict(), osp.join(args.output_dir_src, "source_vitS-IN1k.pth"))
            print("Finish training. Source model saved at "+osp.join(args.output_dir_src, "source_vitS-IN1k.pth"))
        if args.model == 'vit_base':
            torch.save(model.state_dict(), osp.join(args.output_dir_src, "source_vitB-IN1k.pth"))
            print("Finish training. Source model saved at "+osp.join(args.output_dir_src, "source_vitB-IN1k.pth"))

    return model

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SOURCE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--max_epoch', type=int, default=5, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='home', choices=['visda2017', 'home', 'domainnet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--tentimes', default=False, action="store_true", help="whether 10x learning rate for head")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight decay")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")

    parser.add_argument('--output_src', type=str, default='source_model')
    parser.add_argument('--model', type=str, default='vit_small', choices=['vit_small, vit_base'])
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--save', default=False, action="store_true")

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-n2-mstd0', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup-active', action='store_true', default=False,
                        help='enable mixup')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    args = parser.parse_args()

    if args.dset == 'home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'visda2017':
        names = ['synthetic', 'real']
        args.class_num = 12
    if args.dset == 'domainnet':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = args.dataset_path
    if args.dset == 'domainnet':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_train.txt'
        args.t_dset_path = []
        args.test_dset_path = []
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t_dset_path.append(folder + args.dset + '/' + names[i] + '_train.txt')
            args.test_dset_path.append(folder + args.dset + '/' + names[i] + '_test.txt')
    if args.dset == 'home':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_' + str(args.class_num) + '.txt'
        args.t_dset_path = []
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t_dset_path.append(folder + args.dset + '/' + names[i] + '_' + str(args.class_num) + '.txt')
        args.test_dset_path = args.t_dset_path
    if args.dset == 'visda2017':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_' + str(args.class_num) + '.txt'
        args.t_dset_path = []
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t_dset_path.append(folder + args.dset + '/' + names[i] + '_' + str(args.class_num) + '.txt')
        args.test_dset_path = args.t_dset_path

    args.output_dir_src = osp.join(args.output_src, args.dset, names[args.s][0].upper()+'-'+args.model)
    if args.dset == 'domainnet':
        args.name_src = names[args.s][0]
        args.name_test = [name[0] for name in names]
    else:
        args.name_src = names[args.s][0].upper()
        args.name_test = [name[0].upper() for name in names]
    args.name_test.remove(args.name_src)

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    print(args)
    train(args)
    
