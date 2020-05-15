from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import trange
import numpy as np

from data_utils import data_load_emb, LoadData
from data_utils.datastruct import DataStruct

#uds xlnet
class SelfThreeDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self,train_path,dev_path,test_path,dataset):
        loaddata = LoadData(train_path,dev_path,test_path)
        a = loaddata.conllu_counter[dataset]
        counter = loaddata.counter_process(a)
        embedding = BertEmbeddings('bert-base-uncased')

        for i in trange(len(counter)):
            counter[i].sentence_emb = tuple(self.Embedding(counter[i].sentence,embedding))
            print(torch.tensor(counter[i].sentence_emb).shape)
            counter[i].index = i
        self.data=counter
        self.len = len(self.data)
        print(dataset,self.len)


    def Embedding(self,text_list, embedding):
        emb_list = []
        # print(text_list)
        text_list = list(map(Sentence, text_list))
        embedding.embed(text_list)
        for tok in text_list:
            for token in tok:
                # print(token)
                emb_list.append(token.embedding.cpu().numpy())

        return np.array(emb_list)

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