import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from conllu import parse_incr
from tqdm import trange


def generate_mask(length_list, total_leng):
    mask = np.zeros(total_leng)
    mask[length_list] = 1
    return mask.astype(np.bool)


def get_uds_data_with_mask(path="../factuality_eng_udewt/it-happened/it-happened_eng_ud1.2_11012016.tsv",
                           trigger=None, y=None, g=None):
    if y is None:
        y = ['Happened', 'Confidence', 'Sentence']
    if trigger is None:
        trigger = ['Pred.Token', 'Pred.Lemma']
    if g is None:
        g = ["Sentence.data", "Sentence.ID"]
    print("--------------get_uds_data_with_mask-ing-----------------------")

    # data1 = pd.read_csv("../unified/uds/factuality_eng_udewt/it-happened/it-happened_eng_ud1.2_07092017.tsv", sep='\t')
    # data2 = pd.read_csv("../unified/uds/factuality_eng_udewt/it-happened/it-happened_eng_ud1.2_11012016.tsv", sep='\t')
    # data1=data1[data1['Keep']==True]
    # data2['Keep'] = True
    # data = pd.concat([data1, data2])


    data = pd.read_csv("../unified/uds/factuality_eng_udewt/it-happened/it-happened_eng_ud1.2_07092017.tsv", sep='\t')
    data = data[data['Is.Predicate']==True][data['Is.Understandable']==True][data['Happened']!='na'][data['Confidence']!='na'][data['Keep']==True]
    # data = data.reset_index(drop=True)
    # data = data[data['Is.Predicate'] == True][data['Is.Understandable'] == True]
    data = data.reset_index(drop=True)

    data['Confidence'] = data['Confidence'].apply(lambda x: 0.0 if x == 'na' else float(x))
    data['Happened'] = data['Happened'].apply(lambda x: -1.0 if x == 'false' else 1.0)
    data['Confidence'] = data.apply(lambda x: x['Confidence'] * x['Happened'] * (3 / 4), axis=1)

    data = data.groupby(['Sentence.ID', 'Pred.Lemma', 'Pred.Token', 'Split']).mean()
    data = data.reset_index()

    data["Sentence"] = data["Sentence.ID"]
    data["Sentence.data"] = data["Sentence.ID"].apply(str.split).apply(lambda x: x[0])
    data["Sentence.ID"] = data["Sentence.ID"].apply(str.split).apply(lambda x: int(x[1]) - 1)
    data = data.sort_values('Split')
    data = data.reset_index(drop=True)
    counts = data['Split'].value_counts()

    # data to make graph link to /factuality_eng_udewt
    data_graph = data[g]

    index = list(counts.keys())
    index = sorted(index)
    values = list(map(counts.get, index))
    end = np.array(values).cumsum()
    begin = end - values

    dev_mask = list(range(begin[0], end[0]))
    test_mask = list(range(begin[1], end[1]))
    train_mask = list(range(begin[2], end[2]))
    length = len(data)

    dev_mask = generate_mask(dev_mask, length)
    test_mask = generate_mask(test_mask, length)
    train_mask = generate_mask(train_mask, length)

    # factuality
    data_confidence = data.loc[:, y]
    # data_confidence['Confidence'] = data_confidence['Confidence'].apply(lambda x: 0.0 if x == 'na' else float(x))
    # data_confidence['Happened'] = data_confidence['Happened'].apply(lambda x: -1.0 if x == 'false' else 1.0)
    # data_confidence['Confidence'] = data_confidence.apply(lambda x: x['Confidence'] * x['Happened'] * (3 / 4), axis=1)

    # data_trigger word and index
    data_trigger = data.loc[:, trigger]
    data_trigger['Pred.Token'] = data_trigger['Pred.Token'].apply(lambda x: int(x) - 1)
    # -1 means index +1 means <bos>
    print("--------------get_uds_data_with_mask-ed-----------------------")
    return data_graph.values, data_confidence['Confidence'].values, data_trigger[
        ['Pred.Lemma', 'Pred.Token']].values, dev_mask, test_mask, train_mask


def get_word_parsing(dev_path=None, test_path=None, train_path=None):
    # 新版本ud 但fac里标记的旧版
    if dev_path is None:
        dev_path = "../unified/uds/UD-EWT/en_ewt-ud-dev.conllu"
    if test_path is None:
        test_path = "../unified/uds/UD-EWT/en_ewt-ud-test.conllu"
    if train_path is None:
        train_path = "../unified/uds/UD-EWT/en_ewt-ud-train.conllu"
    print("-----------OpenUdConllu------------")
    dev = open(dev_path, "r", encoding="utf-8")
    test = open(test_path, "r", encoding="utf-8")
    train = open(train_path, "r", encoding="utf-8")
    dev_data = []
    test_data = []
    train_data = []
    print("-----------OpenUd---dev---------")
    for tokenlist in parse_incr(dev):
        dev_data.append(tokenlist)
    print("-----------OpenUd---test---------")
    for tokenlist in parse_incr(test):
        test_data.append(tokenlist)
    print("-----------OpenUd---train---------")
    for tokenlist in parse_incr(train):
        train_data.append(tokenlist)

    ud_counter = defaultdict(Counter)
    ud_counter['en-ud-dev.conllu'] = dev_data
    ud_counter['en-ud-test.conllu'] = test_data
    ud_counter['en-ud-train.conllu'] = train_data

    return ud_counter


def get_graph_self(data_graph, ud_counter):
    print("-----------GetGraph-self-----------")
    text_list = []
    edge_index_list = []
    for i in trange(len(data_graph)):
        data = data_graph[i]
        in_edge = []  # in edge without root
        out_edge = []  # out edge
        data = ud_counter.get(data[0])[data[1]]
        text = []

        for j in range(len(data)):
            text.append(data[j]['form'])
            if data[j]['head'] is not None:
                if data[j]['head'] == 0:
                    continue
                in_edge.append(j)
                out_edge.append(data[j]['head'] - 1)


        edge_index_in = in_edge
        edge_index_out = out_edge
        if len(text) == 1 and len(edge_index_in) == 0:
            edge_index_in.append(0)
            edge_index_out.append(0)

        assert len(in_edge) == len(out_edge)
        assert np.array(in_edge).max() < len(text)
        assert np.array(out_edge).max() < len(text)
        assert np.array(out_edge).min() >= 0

        text_list.append(text)
        edge_index_list.append([edge_index_in, edge_index_out])

    return text_list, edge_index_list


def data_loader_api():
    data_graph, data_confidence, data_trigger, dev_mask, test_mask, train_mask = get_uds_data_with_mask()
    ud_counter = get_word_parsing()
    text_list, edge_index_list = get_graph_self(data_graph, ud_counter)
    print(data_trigger)
    # indexss = data_trigger[:, 1]
    # triggers= data_trigger[:,0]
    # list_t = []
    # for i in range(0, len(text_list)):
    #     print("-----------------------------")
    #     print(text_list[i])
    #     print("len_text",len(text_list[i]))
    #     print("index_trigger",indexss[i])
    #     print("trigger",triggers[i])
    #     print("trigger compare",text_list[i][indexss[i]],triggers[i])
    #     list_t.append(text_list[i][indexss[i]])
    # print(list_t)
    # print(data_trigger[:, 0])
    return np.array(text_list), np.array(
        edge_index_list), data_confidence, np.array(test_mask), np.array(
        dev_mask), np.array(train_mask), data_trigger[:, 1]


def data_save():
    text_list, edge_index_list, data_confidence, test_mask, dev_mask, train_mask, data_trigger_index = data_loader_api()
    np.save("../unified/uds/uds/text_list_self.npy", text_list)
    np.save("../unified/uds/uds/edge_index_list_self.npy", edge_index_list)
    np.save("../unified/uds/uds/data_confidence_self.npy", data_confidence)
    np.save("../unified/uds/uds/test_mask_self.npy", test_mask)
    np.save("../unified/uds/uds/dev_mask_self.npy", dev_mask)
    np.save("../unified/uds/uds/train_mask_self.npy", train_mask)
    np.save("../unified/uds/uds/data_trigger_index_self.npy", data_trigger_index)


def data_load():
    text_list = np.load("./unified/uds/uds/text_list_self.npy", allow_pickle=True)
    edge_index_list = np.load("./unified/uds/uds/edge_index_list_self.npy", allow_pickle=True)
    data_confidence = np.load("./unified/uds/uds/data_confidence_self.npy")
    test_mask = np.load("./unified/uds/uds/test_mask_self.npy")
    dev_mask = np.load("./unified/uds/uds/dev_mask_self.npy")
    train_mask = np.load("./unified/uds/uds/train_mask_self.npy")
    data_trigger_index = np.load("./unified/uds/uds/data_trigger_index_self.npy", allow_pickle=True)
    return text_list, edge_index_list, data_confidence, test_mask, dev_mask, train_mask, data_trigger_index

def data_load_emb():
    text_list = np.load("./unified/uds/uds_bert/text_list_self.npy", allow_pickle=True)
    text_list_emb = np.load("./unified/uds/uds_bert/text_list_emb_self.npy", allow_pickle=True)
    edge_index_list = np.load("./unified/uds/uds_bert/edge_index_list_self.npy", allow_pickle=True)
    data_confidence = np.load("./unified/uds/uds_bert/data_confidence_self.npy")
    test_mask = np.load("./unified/uds/uds_bert/test_mask_self.npy")
    dev_mask = np.load("./unified/uds/uds_bert/dev_mask_self.npy")
    train_mask = np.load("./unified/uds/uds_bert/train_mask_self.npy")
    data_trigger_index = np.load("./unified/uds/uds_bert/data_trigger_index_self.npy", allow_pickle=True)

    # text_list = np.load("./unified/uds/uds_xlnet/text_list_self.npy", allow_pickle=True)
    # text_list_emb = np.load("./unified/uds/uds_xlnet/text_list_emb_self.npy", allow_pickle=True)
    # edge_index_list = np.load("./unified/uds/uds_xlnet/edge_index_list_self.npy", allow_pickle=True)
    # data_confidence = np.load("./unified/uds/uds_xlnet/data_confidence_self.npy")
    # test_mask = np.load("./unified/uds/uds_xlnet/test_mask_self.npy")
    # dev_mask = np.load("./unified/uds/uds_xlnet/dev_mask_self.npy")
    # train_mask = np.load("./unified/uds/uds_xlnet/train_mask_self.npy")
    # data_trigger_index = np.load("./unified/uds/uds_xlnet/data_trigger_index_self.npy", allow_pickle=True)
    return text_list, text_list_emb, edge_index_list, data_confidence, test_mask, dev_mask, train_mask, data_trigger_index
if __name__ =='__main__':
    data_save()
