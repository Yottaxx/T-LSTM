import torch
import os

from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Model import Baseline_xlnet_re
import torch
import torch.optim as optim
import torch.nn.functional as F
import json
import time
from conllu import parse_incr
from collections import defaultdict, Counter
import pandas as pd
from tqdm import trange
import numpy as np
import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init

from Model.Sp_lstm.SModelBert import SModelBert
from S_lstm.SSkipModel import SSkipModel
from data_utils.data_loader_three_xlnet import SelfThreeDataset
from data_utils.data_loader_three_xlnet_trans import SelfThreeDataset_trans
from data_utils.data_loader_uds_xlnet_trans import SelfDataset_uds_bert
from data_utils.load_data import LoadData
from data_utils.load_uds import S_get_g_data_loader_split
from data_utils import SelfDataset

torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

# torch.backends.cudnn.deterministic = True

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('tensorboard/baseline/smoothl1/threelss3tm1sssss3512large3')


# fileObject = open('./jsonFile.json', 'w')
# for dict_line in counter:
#     print(dict(dict_line))
#     jsObj = json.dumps(dict(dict_line))
#     print(jsObj)
#     fileObject.write(jsObj)
#
# fileObject.close()

def tokenizer(text):
    return [tok for tok in text]


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


resume = 0
start_epoch = 0
epoch = 2048
train_batch_size = 32
dev_batch_size = 128
test_batch_size = 128

# # three golve without trans
# train_dataset = SelfThreeDataset_trans("./unified/factbank_v1/train.conll", "./unified/factbank_v1/dev.conll", "./unified/factbank_v1/test.conll",'train')
# dev_dataset = SelfThreeDataset_trans("./unified/factbank_v1/train.conll", "./unified/factbank_v1/dev.conll", "./unified/factbank_v1/test.conll",'dev')
# test_dataset = SelfThreeDataset_trans("./unified/factbank_v1/train.conll", "./unified/factbank_v1/dev.conll", "./unified/factbank_v1/test.conll",'test')

#three bert with out bert
train_dataset = SelfDataset_uds_bert('train')
dev_dataset = SelfDataset_uds_bert('dev')
test_dataset = SelfDataset_uds_bert('test')

train_iter = DataLoader(dataset=train_dataset,
                        batch_size=train_batch_size,
                        shuffle=True)
dev_iter = DataLoader(dataset=dev_dataset,
                      batch_size=dev_batch_size,
                      shuffle=True, drop_last=True)
test_iter = DataLoader(dataset=test_dataset,
                       batch_size=test_batch_size,
                       shuffle=False, drop_last=True)

print()

model = SSkipModel(1024,dropout=0.3)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)

# pad_emb = np.load("./unified/uds/uds_bert/pad_emb.npy", allow_pickle=True)
pad_emb=np.zeros((1,1024))


def trainer_train(epoch):
    best_acc = torch.tensor([0.95]).to(device)
    for cycle in trange(epoch):
        temp_loss = 0.0
        temp_mse = 0.0
        count = 0

        for text, text_emb, index, trigger_index, eep in train_iter:
            model.train()
            x = []
            x_mask = []
            max_len = 0

            for ind in index:
                if len(train_dataset.data[ind].sentence) > max_len:
                    max_len = len(train_dataset.data[ind].sentence)

            mode = torch.tensor([pad_emb[0] for _ in range(max_len)], dtype=torch.float)
            mode_mask = torch.zeros(max_len)
            adj = []
            for ind in index:
                temp_mask = mode_mask.clone()
                temp_mask[:len(train_dataset.data[ind].sentence)] = 1
                temp = mode.clone()
                temp[:len(train_dataset.data[ind].sentence_emb)] = torch.tensor(train_dataset.data[ind].sentence_emb)
                x.append(temp)
                x_mask.append(temp_mask)
                adj.append(train_dataset.data[ind].trans_data(max_len))

            x = torch.stack(x, 0).to(device)
            x_mask = torch.stack(x_mask, 0).unsqueeze(-2).to(device)
            trigger = trigger_index.flatten().to(device)
            adj = torch.stack(adj, 0).to(device).to_dense()
            eep = eep.squeeze().to(device)
            optimizer.zero_grad()
            out = model(x, adj, trigger, x_mask)
            # out = model(x,trigger,adj)
            loss = F.smooth_l1_loss(out, eep)
            accu = F.l1_loss(out, eep)
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
            temp_mse += accu.item()
            count += 1
            if count % 50 == 0:
                test_loss = trainer_test(1)
                if test_loss < best_acc:
                    best_acc = test_loss
                    model_path = os.path.join('/media/user1/325655435655094D/baseline/checkpoint/uds',
                                              'lstm1.pth.tar')
                    torch.save({
                        'epoch': cycle,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_acc,
                    }, model_path)

        print('train', (temp_loss / count))
        writer.add_scalar('training loss_baseline_uds',
                          (temp_loss / count),
                          cycle + start_epoch)
        writer.add_scalar('training mse_baseline_uds',
                          (temp_mse / count),
                          cycle + start_epoch)

        dev_loss = trainer_dev(1)
        writer.add_scalar('dev mse_baseline_new',
                          dev_loss,
                          cycle + start_epoch)
        test_loss = trainer_test(1)
        writer.add_scalar('test mse_baseline_new',
                          test_loss,
                          cycle + start_epoch)
        if test_loss < best_acc:
            best_acc = test_loss
            model_path = os.path.join('/media/user1/325655435655094D/baseline/checkpoint/uds',
                                      'lstm1.pth.tar')
            torch.save({
                'epoch': cycle,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_acc,
            }, model_path)


def trainer_dev(epoch):
    loss_list = 0.0
    for cycle in trange(epoch):
        temp_loss = 0.0
        temp_mse = 0.0
        count = 0
        model.eval()
        for text, text_emb, index, trigger_index, eep in dev_iter:

            x = []
            x_mask = []
            max_len = 0

            for ind in index:
                if len(dev_dataset.data[ind].sentence) > max_len:
                    max_len = len(dev_dataset.data[ind].sentence)

            mode = torch.tensor([pad_emb[0] for _ in range(max_len)], dtype=torch.float)
            mode_mask = torch.zeros(max_len)
            adj = []
            for ind in index:
                temp_mask = mode_mask.clone()
                temp_mask[:len(dev_dataset.data[ind].sentence)] = 1
                temp = mode.clone()
                temp[:len(dev_dataset.data[ind].sentence_emb)] = torch.tensor(dev_dataset.data[ind].sentence_emb)
                x.append(temp)
                x_mask.append(temp_mask)
                adj.append(dev_dataset.data[ind].trans_data(max_len))

            x = torch.stack(x, 0).to(device)
            x_mask = torch.stack(x_mask, 0).unsqueeze(-2).to(device)
            trigger = trigger_index.flatten().to(device)
            adj = torch.stack(adj, 0).to(device).to_dense()
            eep = eep.squeeze().to(device)
            # optimizer.zero_grad()
            out = model(x, adj, trigger, x_mask)
            # loss = F.smooth_l1_loss(out, eep)
            accu = F.l1_loss(out, eep)
            # loss.backward()
            # optimizer.step()
            temp_loss += accu.item()
            count += 1
        # print("dev loss:",(temp_loss / count))
        loss_list += (temp_loss / count)
        return (loss_list / epoch)


def trainer_test(epoch):
    loss_list = 0.0
    for cycle in trange(epoch):
        temp_loss = 0.0
        temp_mse = 0.0
        count = 0
        model.eval()
        eval_history_out = []
        eval_history_label = []
        for text, text_emb, index, trigger_index, eep in test_iter:

            x = []
            x_mask = []
            max_len = 0

            for ind in index:
                if len(test_dataset.data[ind].sentence) > max_len:
                    max_len = len(test_dataset.data[ind].sentence)

            mode = torch.tensor([pad_emb[0] for _ in range(max_len)], dtype=torch.float)
            mode_mask = torch.zeros(max_len)
            adj = []
            for ind in index:
                temp_mask = mode_mask.clone()
                temp_mask[:len(test_dataset.data[ind].sentence)] = 1
                temp = mode.clone()
                temp[:len(test_dataset.data[ind].sentence_emb)] = torch.tensor(test_dataset.data[ind].sentence_emb)
                x.append(temp)
                x_mask.append(temp_mask)
                adj.append(test_dataset.data[ind].trans_data(max_len))

            x = torch.stack(x, 0).to(device)
            x_mask = torch.stack(x_mask, 0).unsqueeze(-2).to(device)
            trigger = trigger_index.flatten().to(device)
            adj = torch.stack(adj, 0).to(device).to_dense()
            eep = eep.squeeze().to(device)
            out = model(x, adj, trigger, x_mask)

            loss = F.l1_loss(out, eep)
            temp_loss += loss.item()
            count += 1

            eval_history_out = eval_history_out + out.cpu().detach().numpy().tolist()
            eval_history_label = eval_history_label + eep.cpu().detach().numpy().tolist()

        # print("dev loss:",(temp_loss / count))
        loss_list += (temp_loss / count)
        r = pearsonr(eval_history_out, eval_history_label)
        print("test",loss_list / epoch)
        print("r", r)
        return (loss_list / epoch)


if resume:  # resume为参数，第一次训练时设为0，中断再训练时设为1

    model_path = os.path.join('/media/user1/325655435655094D/baseline/checkpoint/uds',
                              'lstm1.pth.tar')
    assert os.path.isfile(model_path)
    # model.load_state_dict(torch.load(model_path))
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Load checkpoint at epoch {}.'.format(start_epoch))
    print('Best accuracy so far {}.'.format(best_acc))

if __name__ == '__main__':
    trainer_train(epoch)
    test_loss = trainer_test(1)
    print(test_loss)
