from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import trange
import numpy as np
from transformers import BertTokenizer, BertModel

from data_utils import data_load_emb
from data_utils.datastruct import DataStruct

#uds xlnet
class SelfDataset_uds_bert(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self,set_name='train'):
        text_list, text_list_emb,edge_index_list, data_confidence, test_mask, dev_mask, train_mask, data_trigger_index = data_load_emb()
        train_list = []
        dev_list = []
        test_list = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-large-uncased").to('cuda')
        for i in trange(len(data_confidence)):
            x = text_list[i]
            # print("----------------")
            # print("edge")
            # print(edge_index_list[i][0])
            # print(edge_index_list[i][1])
            # print(x)

            edge = np.stack([edge_index_list[i][0], edge_index_list[i][1]], 0)
            #
            # print(len(x))
            edge_index = torch.sparse_coo_tensor(torch.tensor(edge), torch.ones(len(edge[0])),
                                                 (len(x), len(x))).to_dense()
            eep = torch.tensor(data_confidence[i]).unsqueeze(0)
            # print(eep)
            trigger = ["uds"]
            trigger_index = torch.tensor(np.array(data_trigger_index[i], dtype=np.int)).unsqueeze(0)
            # print(x[data_trigger_index[i]])
            ids = torch.tensor(tokenizer.encode(text_list[i])).unsqueeze(0).to('cuda')
            with torch.no_grad():
                text_list_emb[i]= tuple(model(ids)[0][0:, 1:-1, :].squeeze(0).cpu().numpy())
                assert len(text_list_emb[i]) == len(text_list[i]), "after emb"

            if test_mask[i] and set_name =='test':
                data = DataStruct(tuple(text_list[i]), edge_index.numpy().tolist(),
                                  tuple(trigger), tuple(trigger_index.numpy().tolist()),
                                  tuple([eep.numpy().tolist()]), tuple([len(test_list)]),tuple(text_list_emb[i]))
                test_list.append(data)
            if train_mask[i] and set_name=='train':
                data = DataStruct(tuple(text_list[i]), edge_index.numpy().tolist(),
                                  tuple(trigger), tuple(trigger_index.numpy().tolist()),
                                  tuple([eep.numpy().tolist()]), tuple([len(train_list)]),tuple(text_list_emb[i]))
                train_list.append(data)
            if dev_mask[i] and set_name=='dev':
                data = DataStruct(tuple(text_list[i]), edge_index.numpy().tolist(),
                                  tuple(trigger), tuple(trigger_index.numpy().tolist()),
                                  tuple([eep.numpy().tolist()]), tuple([len(dev_list)]),tuple(text_list_emb[i]))
                dev_list.append(data)
        self.train = train_list
        self.dev = dev_list
        self.test = test_list
        if set_name =='dev':
            self .data = dev_list
        if set_name == 'test':
            self.data = test_list
        if set_name =='train':
            self.data = train_list
        for d in range(len(self.data)):
            self.data[d].index=d
        self.len = len(self.data)
    def __getitem__(self, index):
        return self.data[index].sentence,self.data[index].sentence_emb,self.data[index].index,torch.tensor(self.data[index].trigger_index),\
               torch.tensor(self.data[index].eep)

    def __len__(self):
        return self.len


# dealDataset = SelfDataset('train')
#
# train_loader2 = DataLoader(dataset=dealDataset,
#                                batch_size=32,
#                                shuffle=True)
# adj,trigger_ind,eep = next(iter(train_loader2))
#
# print(adj)
# print(trigger_ind)
# print(eep)
#
# print(dealDataset.data[adj[0]].trans_data(100))