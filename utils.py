import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def label_refine(loader, model, args, mode='energy-classwise', iters=0): #modes: energy (default), entropy, confidence
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            outputs, feas = model(inputs, 'Cs-Fs') # obtain prediction via source-oriented function
            if start_test:
                all_fea = feas.float()
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float()), 0)
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = torch.nn.Softmax(dim=1)(all_output)
    conf, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    energy = -torch.log(torch.sum(torch.exp(all_output), dim=1))
    K = all_output.size(1)
    if mode=='energy':
        energy_mask = (energy > energy.mean()).int()
    elif mode=='entropy':
        energy_mask = (ent > ent.mean()).int()
    elif mode=='energy-classwise':
        energy_mask = torch.zeros(energy.size(0)).cuda()
        # print(energy_mask.size())
        for k in range(K):
            cls_mask = (predict==k)
            # print(cls_mask.int().sum())
            cls_energy_mean = (energy*cls_mask.int()).sum()/cls_mask.int().sum()
            # print(cls_energy_mean)
            cls_energy_mask = (energy>cls_energy_mean) * cls_mask
            energy_mask += cls_energy_mask.int()
            # print(k, cls_energy_mask.int().sum())
    elif mode=='entropy-classwise':
        energy_mask = torch.zeros(energy.size(0)).cuda()
        # print(energy_mask.size())
        for k in range(K):
            cls_mask = (predict==k)
            # print(cls_mask.int().sum())
            cls_ent_mean = (ent*cls_mask.int()).sum()/cls_mask.int().sum()
            # print(cls_energy_mean)
            cls_energy_mask = (ent>cls_ent_mean) * cls_mask
            energy_mask += cls_energy_mask.int()
    elif mode=='confidence-classwise':
        energy_mask = torch.zeros(energy.size(0)).cuda()
        # print(energy_mask.size())
        for k in range(K):
            cls_mask = (predict==k)
            # print(cls_mask.int().sum())
            cls_energy_mean = (conf*cls_mask.int()).sum()/cls_mask.int().sum()
            # print(cls_energy_mean)
            cls_energy_mask = (conf<cls_energy_mean) * cls_mask
            energy_mask += cls_energy_mask.int()
    energy_mask_rev = 1-energy_mask
    total = energy.size(0)
    above = torch.sum(energy_mask).item()
    above_ratio = above / total

    reliable_id = torch.where(energy_mask_rev>0)
    unreliable_id = torch.where(energy_mask>0)
    reliable_id = reliable_id[0]
    unreliable_id = unreliable_id[0]

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label.cuda()).item() / float(all_label.size()[0])
    accuracy_above_eng = torch.sum((torch.squeeze(predict).float() == all_label.cuda())*energy_mask).item() / torch.sum(energy_mask).item()
    accuracy_below_eng = torch.sum((torch.squeeze(predict).float() == all_label.cuda())*energy_mask_rev).item() / torch.sum(energy_mask_rev).item()
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).cuda()), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    cls_count = torch.eye(K)[predict.long()].sum(dim=0)

    def cos_pairwise(x1, x2): # pairwise cosine similarity alone dim 1
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        cos_sim = F.cosine_similarity(x1, x2.unsqueeze(1), dim=-2)
        cos_sim = cos_sim.squeeze(-1)
        return 1-cos_sim.T

    reliable_features = all_fea[reliable_id]
    reliable_pred = predict[reliable_id]
    reliable_pred_mask = torch.eye(K)[reliable_pred].cuda()
    reliable_centers = reliable_features.t() @ reliable_pred_mask
    reliable_centers = (reliable_centers / (1e-8 + reliable_pred_mask.sum(dim=0))).t()
    print("reliable pred numbers: ", reliable_pred_mask.sum(dim=0))
    unreliable_features = all_fea[unreliable_id]

    balance_weight = torch.exp(reliable_pred_mask.sum(dim=0)/reliable_pred_mask.sum())

    if args.dset=='domainnet':
        dd = cos_pairwise(unreliable_features.cpu(), reliable_centers.cpu())
        dd = dd*balance_weight.cpu()
    else:
        dd = cos_pairwise(unreliable_features, reliable_centers)
        dd = dd*balance_weight

    cloest = dd.argmin(axis=1)
    pred_label = predict.cpu()
    pred_label[unreliable_id] = cloest.cpu()
    pred_label = pred_label

    accur = torch.sum(pred_label == all_label).item() / float(all_label.size()[0])
    accur_above_eng = torch.sum((pred_label == all_label).cuda()*energy_mask.cuda()).item() / torch.sum(energy_mask).item()
    accur_below_eng = torch.sum((pred_label == all_label).cuda()*energy_mask_rev.cuda()).item() / torch.sum(energy_mask_rev).item()

    log_str = 'Accuracy [all,unreliable,reliable] = [{:.2f}%, {:.2f}%, {:.2f}%] -> [{:.2f}%, {:.2f}%, {:.2f}%]\n'.format(accuracy*100, accuracy_above_eng*100, accuracy_below_eng*100, accur * 100, accur_above_eng*100, accur_below_eng*100)
    log_str += 'unreliable/reliable ratio: {:.1f}/{:.1f}'.format(above_ratio*100, (1-above_ratio)*100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return pred_label

def select_pseudo_label_by_threshold(pseudo_loader, model, pseudo_file, args):
    model.eval()
    print("write pseudo label file to: [" , pseudo_file,"]")
    pseudo = open(pseudo_file, "w")
    with torch.no_grad():
        correct = 0
        pseudo_size = 0
        total_size = 0
        for i, (inputs, label, path) in enumerate(pseudo_loader):
            inputs = inputs.cuda()
            label = label.cuda()
            outputs, _ = model(inputs, 'pseudo')

            predict_out = nn.Softmax()(outputs)
            confidence, predict = torch.max(predict_out,1)

            for idx, conf in enumerate(confidence):
                if conf.item() > args.pseudo_threshold:
                    pseudo.flush()
                    pseudo.write(path[idx] + " ")
                    pseudo.write("{:d}\n".format(predict[idx].item()))
                    pseudo_size += 1
                    if predict[idx].item() == label[idx].item():
                        correct += 1
                total_size += 1

    pseudo.close()
    model.train()
    print('Pseudo label setected: {:.2f}[{}/{}],  Accuracy: {:.2f}%'.format(pseudo_size/total_size, pseudo_size, total_size, correct/pseudo_size))
    return pseudo_file

import heapq
class MinHeap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self._data)[1]

    def __len__(self):
        return len(self._data)

def select_pseudo_label_by_CBST(pseudo_loader, model, pseudo_file, portion, args):
    model.eval()
    print("write pseudo label file to: [" , pseudo_file,"]")
    pseudo = open(pseudo_file, "w")
    distance_heap = MinHeap(key=lambda item: item[2])
    class_pred_num = [0] * args.class_num
    total_size = 0
    correct_size = 0
    with torch.no_grad():
        for i, (inputs, label, path) in enumerate(pseudo_loader):
            inputs = inputs.cuda()
            label = label.cuda()
            outputs, _ = model(inputs, 'pseudo')

            predict_out = nn.Softmax()(outputs)
            confidence, predict = torch.max(predict_out,1)

            for idx, conf in enumerate(confidence):
                class_pred_num[predict[idx].item()] += 1
                correct = int(predict[idx].item()==label[idx].item())
                distance_heap.push([path[idx], predict[idx].item(), -conf.item(),correct])
                total_size += 1
                if correct>0:
                    correct_size+=1

    print("correct:",correct_size)

    print("cpn:",class_pred_num)
    class_pl_num = [0]*args.class_num
    for i, pred_num in enumerate(class_pred_num):
        class_pl_num[i] = math.floor(portion*pred_num)

    print("cpln:",class_pl_num)
    running_class_pl_num = [0]*args.class_num
    pseudo_correct = 0
    pseudo_size = 0
    for i in range(len(distance_heap)):
        data = distance_heap.pop()
        path, cls_idx, correct = data[0], data[1], data[3]
        if running_class_pl_num[cls_idx] <= class_pl_num[cls_idx]:
            running_class_pl_num[cls_idx] += 1
            pseudo.flush()
            pseudo.write(path + " ")
            pseudo.write("{:d}\n".format(cls_idx))
            pseudo_size += 1
            if correct>0:
                pseudo_correct += 1

    pseudo.close()
    model.train()
    print('Pseudo label setected: {:.2f}[{}/{}],  Accuracy: {:.2f}%'.format(pseudo_size/total_size, pseudo_size, total_size, pseudo_correct/pseudo_size))
    return pseudo_file

def select_pseudo_label_by_nacl(src_loader, tgt_loader, model, pseudo_labeler, args):
    model.eval()

    class_pred_num = [0] * args.class_num
    with torch.no_grad():
        src_feat = []
        tgt_feat = []
        all_src_label = []
        all_tgt_label = []
        for i, (inputs, label) in enumerate(src_loader):
            inputs = inputs.cuda()
            label = label.cuda()
            outputs, features = model(inputs, 'pseudo')
            src_feat.append(features)
            all_src_label.append(label)

        for i, (inputs, label) in enumerate(tgt_loader):
            inputs = inputs.cuda()
            label = label.cuda()
            outputs, features = model(inputs, 'pseudo')
            tgt_feat.append(features)
            all_tgt_label.append(label)
    
    src_feat = torch.cat(tuple(src_feat), dim=0)
    tgt_feat = torch.cat(tuple(tgt_feat), dim=0)
    all_src_label = torch.cat(tuple(all_src_label), dim=0).float()
    all_tgt_label = torch.cat(tuple(all_tgt_label), dim=0).float()

    src_collection = {'features':src_feat, 'true_labels':all_src_label}
    tgt_collection = {'features':tgt_feat, 'true_labels':all_tgt_label}
    pseudo_label = pseudo_labeler.pseudo_label_tgt(src_collection, tgt_collection)
    pseudo_label = pseudo_label.max(dim=1)[1]
    correct = (pseudo_label==all_tgt_label).sum()
    print(pseudo_label[0],all_tgt_label[0])

    model.train()
    print('Pseudo label accuracy: {:.2f}%'.format(correct/all_tgt_label.size(0)))
    return pseudo_label

