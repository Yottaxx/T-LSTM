import json

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
from torchtext.data import Example, Iterator
from data_utils.datastruct import DataStruct

#meantime and uw torchtext glove
class LoadData:
    """ """

    def __init__(self, train_path, dev_path, test_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.conllu_counter = self.get_conllu()
        self.example = Example()

        def tokenizer(text):
            return [tok for tok in text]

        self.text = data.Field(sequential=True, tokenize=tokenizer, lower=True)

    def get_conllu(self):
        print("-----------OpenUdConllu------------")
        dev = pd.read_csv(self.dev_path, sep='\t',
                          names=['index', 'token', 'eep', 'pos1', 'head', 'pos2', 'original_word'])
        test = pd.read_csv(self.test_path, sep='\t',
                           names=['index', 'token', 'eep', 'pos1', 'head', 'pos2', 'original_word'])
        train = pd.read_csv(self.train_path, sep='\t',
                            names=['index', 'token', 'eep', 'pos1', 'head', 'pos2', 'original_word'])

        ud_counter = defaultdict(Counter)
        ud_counter['dev'] = dev
        ud_counter['test'] = test
        ud_counter['train'] = train
        return ud_counter

    def process(self, text):
        total = []
        total_trigger = []
        total_trigger_index = []
        total_edgei = []
        total_edgej = []
        total_adj = []
        temp_line_trigger = []
        temp_line_trigger_index = []
        temp_line = []
        temp_line_edgei = []
        temp_line_edgej = []
        count = 0
        for i in trange(len(text)):
            line = text.loc[i]
            if line['index'] == 0 and len(temp_line) != 0:
                count += 1
                for j in range(len(temp_line_trigger)):
                    total.append(temp_line.copy())
                for j in range(len(temp_line_trigger)):
                    total_edgei.append(temp_line_edgei.copy())
                    total_edgej.append(temp_line_edgej.copy())

                # bound condition
                assert ((np.array(temp_line_edgej)).max()) < len(temp_line)
                assert ((np.array(temp_line_edgei)).max()) < len(temp_line)

                total_trigger += temp_line_trigger
                total_trigger_index += temp_line_trigger_index
                temp_line.clear()
                temp_line_trigger.clear()
                temp_line_trigger_index.clear()
                temp_line_edgei.clear()
                temp_line_edgej.clear()

                if count == 4:
                    break

            temp_line.append(line['token'].lower())
            if line['eep'] != '_':
                temp_line_trigger.append(line['token'].lower())
                temp_line_trigger_index.append(line['index'])

            if line['head'] >= 0:
                temp_line_edgei.append(line['index'])
                temp_line_edgej.append(line['head'])

        if len(temp_line) != 0:
            for _ in range(len(temp_line_trigger_index)):
                total.append(temp_line)
            total_trigger += temp_line_trigger
            total_trigger_index += temp_line_trigger_index
            temp_line.clear()
            temp_line_trigger.clear()
            temp_line_trigger_index.clear()

        return total

    def counter_process(self, text):
        total = []
        temp_line_trigger = []
        temp_line_trigger_index = []
        temp_line = []
        temp_line_edgei = []
        temp_line_edgej = []
        temp_line_eep = []
        for i in trange(len(text)):
            line = text.loc[i]
            if line['index'] == 0 and len(temp_line) != 0:
                for j in range(len(temp_line_trigger)):
                    adj = torch.sparse_coo_tensor(torch.cat((torch.tensor(temp_line_edgei.copy()).unsqueeze(0),
                                                             torch.tensor(temp_line_edgej.copy(),dtype=torch.float).unsqueeze(0)), 0),
                                                  torch.ones(len(temp_line_edgei)),
                                                  (len(temp_line), len(temp_line))).to_dense().numpy().tolist()
                    data_line = DataStruct(tuple(temp_line.copy()), adj,
                                           tuple([temp_line_trigger[j]]), tuple([temp_line_trigger_index[j]]),
                                           tuple([temp_line_eep[j]]), tuple([len(total)]))

                    total.append(data_line)
                    assert data_line.sentence[int(data_line.trigger_index[0])] == data_line.trigger[
                        0], 'trigger_matching'

                    # bound condition
                    assert ((np.array(temp_line_edgej)).max()) < len(temp_line)
                    assert ((np.array(temp_line_edgei)).max()) < len(temp_line)

                temp_line.clear()
                temp_line_trigger.clear()
                temp_line_trigger_index.clear()
                temp_line_edgei.clear()
                temp_line_edgej.clear()
                temp_line_eep.clear()

            temp_line.append(line['token'].lower())
            if line['eep'] != '_' and line['eep'] != 'nan':
                if -3 <= float(line['eep']) <= 3:
                    temp_line_trigger.append(line['token'].lower())
                    temp_line_trigger_index.append(float(line['index']))
                    temp_line_eep.append(float(line['eep']))

            if line['head'] >= 0:
                temp_line_edgei.append(float(line['index']))
                temp_line_edgej.append(line['head'])

        if len(temp_line) != 0:
            for j in range(len(temp_line_trigger)):
                adj = torch.sparse_coo_tensor(torch.cat((torch.tensor(temp_line_edgei.copy()).unsqueeze(0),
                                                         torch.tensor(temp_line_edgej.copy(),dtype=torch.float).unsqueeze(0)), 0),
                                              torch.ones(len(temp_line_edgei)),
                                              (len(temp_line), len(temp_line))).to_dense().numpy().tolist()
                data_line = DataStruct(tuple(temp_line.copy()), adj,
                                       tuple([temp_line_trigger[j]]), tuple([temp_line_trigger_index[j]]),
                                       tuple([temp_line_eep[j]]), tuple([len(total)]))

                total.append(data_line)
                assert data_line.sentence[int(data_line.trigger_index[0])] == data_line.trigger[0], 'trigger_matching'

                assert ((np.array(temp_line_edgej)).max()) < len(temp_line)
                assert ((np.array(temp_line_edgei)).max()) < len(temp_line)

        return total


if __name__ == "__main__":
    loaddata = LoadData("../unified/uw/train.conll", "../unified/uw/dev.conll", "../unified/uw/test.conll")
    a = loaddata.conllu_counter['train']
    counter = loaddata.counter_process(a)
    b = loaddata.conllu_counter['dev']
    counter_dev = loaddata.counter_process(b)
    c = loaddata.conllu_counter['test']
    counter_test = loaddata.counter_process(c)

    for_vocab = []
    for_vocab = for_vocab + counter + counter_test + counter_dev


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


    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    ADJ = data.Field(sequential=False, use_vocab=False)
    TRIGGERINDEX = data.Field(sequential=False, use_vocab=False)
    EEP = data.Field(sequential=False, use_vocab=False, dtype=torch.float32)
    INDEX = data.Field(sequential=False, use_vocab=False)

    train = data.Dataset(counter, fields=[('sentence', TEXT),
                                          ('adj', None), ('trigger', TEXT), ('trigger_index', TRIGGERINDEX),
                                          ('eep', EEP), ('index', INDEX)])

    dev = data.Dataset(counter_dev, fields=[('sentence', TEXT),
                                            ('adj', None), ('trigger', TEXT), ('trigger_index', TRIGGERINDEX),
                                            ('eep', EEP), ('index', INDEX)])

    test = data.Dataset(counter_test, fields=[('sentence', TEXT),
                                              ('adj', None), ('trigger', TEXT), ('trigger_index', TRIGGERINDEX),
                                              ('eep', EEP), ('index', INDEX)])

    for_vocab = data.Dataset(for_vocab, fields=[('sentence', TEXT),
                                                ('adj', None), ('trigger', None), ('trigger_index', None),
                                                ('eep', None), ('index', None)])

    TEXT.build_vocab(for_vocab, vectors='glove.6B.100d')  # , max_size=30000)
    TEXT.vocab.vectors.unk_init = init.xavier_uniform
    print(TEXT.vocab.vectors.shape)
    print()

    train_iter = data.BucketIterator(train, batch_size=64, train=True,
                                     sort_within_batch=True,
                                     sort_key=lambda x: (len(x.sentence)), repeat=False,
                                     device='cuda')

    for batch in train_iter:
        print(batch)
        for i in batch.index:
            print(len(counter[i].sentence))
            print(batch.sentence.shape[0])
            assert len(counter[i].sentence) <= batch.sentence.shape[0]

        x = batch.sentence.t()
        adj = []
        trigger = batch.trigger_index.t().flatten()

        count = 0
        for ind in batch.index:
            adj.append(counter[ind].trans_data(x.shape[-1]))
        adj = torch.stack(adj, 0)
        mask = get_pad_mask(x, TEXT.vocab.stoi[TEXT.pad_token])
        eep = batch.eep.squeeze()
        print(x.shape)
        print(adj.shape)
        print(trigger.shape)
        print(mask.shape)
