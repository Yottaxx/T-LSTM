from data_utils import DataStruct
from tqdm import trange
import torch
from data_utils.save_uds_utils import data_load
import numpy as np


def S_get_g_data_loader_split():
    text_list, edge_index_list, data_confidence, test_mask, dev_mask, train_mask, data_trigger_index = data_load()
    train_list = []
    dev_list = []
    test_list = []
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
        if test_mask[i] :
            data = DataStruct(tuple(text_list[i]), edge_index.numpy().tolist(),
                              tuple(trigger), tuple(trigger_index.numpy().tolist()),
                              tuple(eep.numpy().tolist()), tuple([len(test_list)]))
            test_list.append(data)
        if train_mask[i]:
            data = DataStruct(tuple(text_list[i]), edge_index.numpy().tolist(),
                              tuple(trigger), tuple(trigger_index.numpy().tolist()),
                              tuple(eep.numpy().tolist()), tuple([len(train_list)]))
            train_list.append(data)
        if dev_mask[i] :
            data = DataStruct(tuple(text_list[i]), edge_index.numpy().tolist(),
                              tuple(trigger), tuple(trigger_index.numpy().tolist()),
                              tuple(eep.numpy().tolist()), tuple([len(dev_list)]))
            dev_list.append(data)

    return train_list, dev_list, test_list

def S_get_g_data_loader_split_xlnet():
    text_list, text_list_emb,edge_index_list, data_confidence, test_mask, dev_mask, train_mask, data_trigger_index = data_load()
    train_list = []
    dev_list = []
    test_list = []
    for i in trange(len(data_confidence)):
        x = text_list[i]
        x_emb = torch.tensor(text_list_emb[i])
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
        if test_mask[i] :
            data = DataStruct(tuple(text_list[i]), x_emb,edge_index.numpy().tolist(),
                              tuple(trigger), tuple(trigger_index.numpy().tolist()),
                              tuple([eep.numpy().tolist()]), tuple([len(test_list)]))
            test_list.append(data)
        if train_mask[i]:
            data = DataStruct(tuple(text_list[i]),x_emb ,edge_index.numpy().tolist(),
                              tuple(trigger), tuple(trigger_index.numpy().tolist()),
                              tuple([eep.numpy().tolist()]), tuple([len(train_list)]))
            train_list.append(data)
        if dev_mask[i] :
            data = DataStruct(tuple(text_list[i]),x_emb ,edge_index.numpy().tolist(),
                              tuple(trigger), tuple(trigger_index.numpy().tolist()),
                              tuple([eep.numpy().tolist()]), tuple([len(dev_list)]))
            dev_list.append(data)

    return train_list, dev_list, test_list