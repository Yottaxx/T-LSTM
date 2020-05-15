from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import trange
import numpy as np

from data_utils import data_load_emb, LoadData
from data_utils.datastruct import DataStruct
from transformers import BertTokenizer,BertModel
#uds xlnet
class SelfThreeDataset_trans(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self,train_path,dev_path,test_path,dataset):
        loaddata = LoadData(train_path,dev_path,test_path)
        a = loaddata.conllu_counter[dataset]
        counter = loaddata.counter_process(a)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased").to('cuda')
        for i in trange(len(counter)):
            ids = torch.tensor(tokenizer.encode(counter[i].sentence)).unsqueeze(0).to('cuda')
            with torch.no_grad():
                counter[i].sentence_emb = tuple(model(ids)[0][0:,1:-1,:].squeeze(0).cpu().numpy())
                assert len(counter[i].sentence_emb) == len(counter[i].sentence),"after emb"
            counter[i].index = i
        self.data=counter
        self.len = len(self.data)
        print(dataset,self.len)


    def __getitem__(self, index):
        return self.data[index].sentence,self.data[index].sentence_emb,self.data[index].index,torch.tensor(self.data[index].trigger_index,dtype=torch.long),\
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