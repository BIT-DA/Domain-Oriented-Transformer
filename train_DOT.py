import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random
from loss import SupConLoss
from sklearn.metrics import confusion_matrix
from timm_diy.models import create_model
from timm_diy.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import label_refine

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
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
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["source"] = ImageList(txt_src, transform=build_transform(True, args))
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dsets["target"] = ImageList_idx(txt_tar, transform=build_transform(True, args))
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dsets["retrieval"] = ImageList(txt_tar, transform=build_transform(False, args))
    dset_loaders["retrieval"] = DataLoader(dsets["retrieval"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker,
                                           drop_last=False)                                   
    dsets["test"] = ImageList(txt_test, transform=build_transform(False, args))
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders

def cal_acc(loader, model, visda=False, mode='Ct-Ft'):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs, _ = model(inputs, mode)
            if start_test:
                all_outputs = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                # all_label2 = torch.cat((all_label2, labels.float()), 0)
    _, predict = torch.max(all_outputs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label.cuda()).item() / float(all_label.size()[0])
    
    if visda:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).cpu().float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, accuracy * 100

def train(args):
    log_str = '{}->{}'.format(args.names[args.s], args.names[args.t])
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print('{}->{}'.format(args.names[args.s], args.names[args.t]))
    dset_loaders = data_load(args)

    if args.model == 'dot_small':
        pre_model = create_model("vit_small_patch16_224", pretrained=False, num_classes=args.class_num)
        modelpath = args.src_model_path + "/source_vitS-IN1k.pth"
        
    elif args.model == 'dot_base':
        pre_model = create_model("vit_base_patch16_224", pretrained=False, num_classes=args.class_num)
        modelpath = args.src_model_path + "/source_vitB-IN1k.pth"
        
    pretrained = torch.load(modelpath)
    pos_embed = pretrained['pos_embed'].data
    # if args.model.startswith('deit'):
    #     del pretrained['head_dist.weight'], pretrained['head_dist.bias'], pretrained['dist_token']
    #     pos_embed = torch.cat([pos_embed[:,0:1],pos_embed[:,2:]],dim=1)
    del pretrained['pos_embed']
    pre_model.load_state_dict(pretrained, strict=False)
    pre_model.pos_embed.data = pos_embed

    pse_label = label_refine(dset_loaders["retrieval"], pre_model.cuda(), args, mode='energy-classwise')
    del pre_model

    # initialize the DOT model
    if args.model=='dot_small':
        print('[[ Backbone: dot_small ]]')
        model = create_model("dot_small_patch16_224", pretrained=False, num_classes=args.class_num)
        pretrained_model = './pretrained/deit_small_distilled_patch16_224-649709d9.pth'
    elif args.model=='dot_base':
        print('[[ Backbone: dot_base ]]')
        model = create_model("dot_base_patch16_224", pretrained=False, num_classes=args.class_num)
        pretrained_model = './pretrained/deit_base_distilled_patch16_224-df68dfff.pth'
    
    print('Initializing with Deit IN-1k pretrained model.')
    pretrained = torch.load(pretrained_model)
    cls_token = pretrained['cls_token']
    dist_token = pretrained['dist_token']
    pos_embed = pretrained['pos_embed']
    
    del pretrained['head.weight'], pretrained['head.bias']
    del pretrained['head_dist.weight'], pretrained['head_dist.bias']

    cls_token, dist_token = model.cls_token.data, model.dist_token.data 
    cls_pos, dist_pos = model.pos_embed.data[:,0].squeeze(), model.pos_embed.data[:,1].squeeze()
    model.cls_token.data = cls_token * cls_pos.norm(p=2)/cls_token.norm(p=2)
    model.dist_token.data = dist_token * dist_pos.norm(p=2)/dist_token.norm(p=2)

    if args.randtoken:
        pretrained['cls_token'] = model.cls_token.data * pos_embed[:,0].squeeze().norm(p=2)/model.cls_token.data.norm(p=2)
        pretrained['dist_token'] = model.dist_token.data * pos_embed[:,1].squeeze().norm(p=2)/model.dist_token.data.norm(p=2)
    else:
        pretrained['cls_token'] = cls_token * pos_embed[:,0].squeeze().norm(p=2)/cls_token.norm(p=2)
        pretrained['dist_token'] = dist_token * pos_embed[:,1].squeeze().norm(p=2)/dist_token.norm(p=2)

    model.load_state_dict(pretrained, strict=False)
    model = model.cuda()
    print("cls_token norm: ", model.cls_token.data.squeeze().norm(p=2).item())
    print("pos_embed norm: ", model.pos_embed.data[:,0].squeeze().norm(p=2).item())

    learning_rate = args.lr
    param_group = []
    for k, v in model.named_parameters():
        if k.find('head') != -1:
            param_group += [{'params': v, 'lr': learning_rate * args.lr_mult}]
        else:
            param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    criterion = nn.CrossEntropyLoss()
    supcon = SupConLoss()

    max_iter = args.max_epoch * args.iter_per_epoch
    interval_iter = args.iter_per_epoch

    print("max-iter : {}".format(max_iter))
    iter_num = 0
    best_acc = 0.0
    best_model={}
    sum_cls_loss = 0.0
    sum_pl_loss = 0.0
    sum_con_loss = 0.0
    sum_feat_loss = 0.0
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except Exception as e:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = iter_source.next()

        try:
            inputs_target, _, idx_target = iter_target.next()
        except Exception as e:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _, idx_target = iter_target.next()

        iter_num += 1
        model.train()
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        inputs_target = inputs_target.cuda()
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        outputs_cls, outputs_dist, features_cls, features_dist = model(inputs)

        features_cls_source, features_cls_target = features_cls.chunk(2, dim=0)
        outputs_cls_source, outputs_cls_target = outputs_cls.chunk(2, dim=0)
        features_dist_source, features_dist_target = features_dist.chunk(2, dim=0)
        outputs_dist_source, outputs_dist_target = outputs_dist.chunk(2, dim=0)

        cls_loss = criterion(outputs_cls_source, labels_source)
        labels_target_pseudo = pse_label[idx_target].cuda()
        pse_loss = criterion(outputs_dist_target, labels_target_pseudo)

        con_loss_s = supcon(features_cls_target, labels_target_pseudo, features_cls_source, labels_source)
        con_loss_t = supcon(features_dist_source, labels_source, features_dist_target, labels_target_pseudo)
        con_loss = con_loss_s + con_loss_t

        disentangle_loss = nn.functional.cosine_similarity(features_cls,features_dist,dim=1).square().mean()

        total_loss = (cls_loss + pse_loss)*args.cls_par + con_loss*args.cons_par + disentangle_loss*args.feat_par

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        sum_cls_loss += cls_loss.item()
        sum_pl_loss += pse_loss.item()
        sum_con_loss += con_loss.item()*args.cons_par
        sum_feat_loss += disentangle_loss.item()*args.feat_par
        if iter_num % 100 == 0:
            log_str = 'Iter: {}, ClsLoss = {:.3f}, PCLSLoss = {:.3f}, ContrastiveLoss = {:.3f}, featLoss:{:.3f}'.format(
                iter_num,
                sum_cls_loss / 100,
                sum_pl_loss / 100,
                sum_con_loss / 100, 
                sum_feat_loss / 100,)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            sum_pl_loss = 0.
            sum_cls_loss = 0.
            sum_con_loss = 0.
            sum_feat_loss = 0
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            torch.cuda.empty_cache()
            if args.dset == 'visda2017':
                acc, acc_list = cal_acc(dset_loaders['test'], model, True, mode='Ct-Ft')
                log_str = '\nCt-Ft-Xt: {}{}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, args.name_tgt, iter_num, max_iter,
                                                                                    acc) + '\n' + acc_list
            else:
                acc, _ = cal_acc(dset_loaders['test'], model, False, mode='Ct-Ft')
                log_str = '\nCt-Ft-Xt: {}{}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, args.name_tgt, iter_num, max_iter,
                                                                                    acc)
            best_acc = max(best_acc, acc)
            best_model = model.state_dict()                                               
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            pse_label = label_refine(dset_loaders['retrieval'], model, args, mode='energy-classwise', iters=iter_num)

    print("Finish training, best accuracy: {:.2f}%".format(best_acc))
    torch.save(best_model, osp.join(args.output_dir, 'best_model.pth'))

    return model


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--iter_per_epoch', type=int, default=500, help="iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='home', choices=['visda2017', 'home', 'domainnet'])
    parser.add_argument('--lr', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--lr_mult', type=float, default=10, help="head learning rate multiplier")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='dot_small', choices=['dot_small', 'dot_base'])
   
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--dataset_path', type=str, default='./data/')
    parser.add_argument("--src_model_path", type=str, default="source_model")
    
    parser.add_argument('--cons_par', type=float, default=1.0)
    parser.add_argument('--cls_par', type=float, default=1.0)
    parser.add_argument('--feat_par', type=float, default=0.1)

    # Augmentation parameter
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
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

    # initialization options
    parser.add_argument('--randtoken', type=int, default=1, choices=[0,1])

    args = parser.parse_args()

    if args.dset == 'home':
        args.names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda2017':
        args.names = ['synthetic', 'real']
        args.class_num = 12
    if args.dset == 'domainnet':
        args.names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # for s in range(len(args.names)):
    folder = args.dataset_path
    if args.dset == 'domainnet':
        args.s_dset_path = folder + args.dset + '/' + args.names[args.s] + '_train.txt'
        args.t_dset_path = folder + args.dset + '/' + args.names[args.t] + '_train.txt'
        args.test_dset_path = folder + args.dset + '/' + args.names[args.t] + '_test.txt'
    else:
        args.s_dset_path = (folder + args.dset + "/" + args.names[args.s] + "_" + str(args.class_num) + ".txt")
        args.t_dset_path = (folder + args.dset + "/" + args.names[args.t] + "_" + str(args.class_num) + ".txt")
        args.test_dset_path = (folder + args.dset + "/" + args.names[args.t] + "_" + str(args.class_num) + ".txt")

    args.output_dir = osp.join(args.output_dir, args.dset, args.names[args.s][0].upper() + args.names[args.t][0].upper())
    src_model = 'vit_small' if args.model == 'dot_small' else 'vit_base'
    args.src_model_path = osp.join(args.src_model_path, args.dset, args.names[args.s][0].upper()+'-'+src_model)
    if args.dset == 'domainnet':
        args.name_src = args.names[args.s][0]
        args.name_tgt = args.names[args.t][0]
    else:
        args.name_src = args.names[args.s][0].upper()
        args.name_tgt = args.names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    print(print_args(args) + '\n')
    train(args)
