import torch
import os

from scipy.stats import pearsonr
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
import torch
from torchtext import data, datasets
from torch.nn import init

from S_lstm.SModel import SModel
from data_utils.load_uds import S_get_g_data_loader_split
from data_utils import SelfDataset
import argparse
import nni
import logging

logger = logging.getLogger('TLSTM_autoML')


# three golve without trans

# train_dataset = SelfDataset_uds_bert('train')
# dev_dataset = SelfDataset_uds_bert('dev')
# test_dataset = SelfDataset_uds_bert('test')
def get_parmas():
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--lr', '-l', type=float, help="lr must", default=0.0001)
    parser.add_argument('--batch_size', '-b', type=int, help="batch_size must", default=32)
    parser.add_argument('--epoch', '-e', type=int, help="epoch must", default=128)
    parser.add_argument('--dropout', '-d', type=float, help="dropout must", default=0.3)
    parser.add_argument('--in_size', '-i', type=int, help="in_size must", default=512)
    parser.add_argument('--g_size', '-g', type=int, help="g_size must", default=300)
    args, _ = parser.parse_known_args()
    return args


def run(args):
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)

    # torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaddata = LoadData("./unified/uw/train.conll", "./unified/uw/dev.conll", "./unified/uw/test.conll")

    counter, counter_dev, counter_test = S_get_g_data_loader_split()

    # a = loaddata.conllu_counter['train']
    # a = loaddata.counter_process(a)
    # for d in a:
    #     counter.append(d)
    # for d in range(len(counter)):
    #     counter[d].index=tuple([d])
    # b = loaddata.conllu_counter['dev']
    # b = loaddata.counter_process(b)
    # for d in b :
    #     counter_dev.append(d)
    # for d in range(len(counter_dev)):
    #     counter_dev[d].index=tuple([d])
    # c = loaddata.conllu_counter['test']
    # c = loaddata.counter_process(c)
    # test_i=c.copy()
    # for d in c:
    #     counter_test.append(d)
    # for d in range(len(counter_test)):
    #     counter_test[d].index=tuple([d])
    print("train length", len(counter))
    print("dev length", len(counter_dev))
    print("test length", len(counter_test))
    for_vocab = []
    for_vocab = for_vocab + counter + counter_test + counter_dev
    #
    # for t in range(len(test_i)):
    #     test_i[t].index=tuple([t])
    # counter_test =test_i
    print("this_test", len(counter_test))

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

    TEXT.build_vocab(for_vocab, vectors='glove.42B.300d')  # , max_size=30000)
    TEXT.vocab.vectors.unk_init = init.xavier_uniform
    print(TEXT.vocab.vectors.shape)
    print()

    resume = 0
    start_epoch = 0
    epoch = args['epoch']
    train_batch_size = args['batch_size']
    dev_batch_size = args['batch_size']
    test_batch_size = args['batch_size']
    train_iter = data.BucketIterator(train, batch_size=train_batch_size, train=True,
                                     sort_within_batch=True,
                                     sort_key=lambda x: (len(x.sentence)), repeat=False,
                                     device='cpu')

    dev_iter = data.BucketIterator(dev, batch_size=dev_batch_size, train=True,
                                   sort_within_batch=True,
                                   sort_key=lambda x: (len(x.sentence)), repeat=False,
                                   device='cpu')

    test_iter = data.BucketIterator(test, batch_size=test_batch_size, train=False,
                                    sort_within_batch=True,
                                    sort_key=lambda x: (len(x.sentence)), repeat=False,
                                    device='cpu')

    len_vocab = len(TEXT.vocab)

    model = SModel(len_vocab, emb_size=300, in_size=args['in_size'], g_size=args['in_size'],
                   linear_h_size=args['in_size'], dropout=args['dropout'])
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), weight_decay=5e-4)

    def trainer_train(epoch):
        best_acc = torch.tensor([2.00]).to(device)
        for cycle in trange(epoch):
            temp_loss = 0.0
            temp_mse = 0.0
            count = 0

            for batch in train_iter:
                model.train()
                for i in batch.index:
                    assert len(counter[i].sentence) <= batch.sentence.shape[0], "graph out-side"

                x = batch.sentence.t().to(device)
                adj = []
                trigger = batch.trigger_index.t().flatten().to(device)

                for ind in batch.index:
                    adj.append(counter[ind].trans_data(x.shape[-1]))
                adj = torch.stack(adj, 0).to(device).to_dense()

                mask = get_pad_mask(x, TEXT.vocab.stoi[TEXT.pad_token]).to(device)
                eep = batch.eep.squeeze().to(device)

                # optim = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
                optimizer.zero_grad()
                out = model(x, trigger, adj)
                loss = F.smooth_l1_loss(out, eep)
                accu = F.l1_loss(out, eep)
                loss.backward()
                optimizer.step()

                temp_loss += loss.item()
                temp_mse += accu.item()
                count += 1
                if count % 50 == 0:
                    test_loss = trainer_test(1)
            dev_loss = trainer_dev(1)

            test_loss = trainer_test(1)

            nni.report_intermediate_result(test_loss)
            logger.debug('dev mae %g', dev_loss)
            logger.debug('test mae %g', test_loss)
            if test_loss < best_acc:
                best_acc = test_loss
        nni.report_final_result(best_acc)

    def trainer_dev(epoch):
        loss_list = 0.0
        for cycle in trange(epoch):
            model.eval()
            temp_loss = 0.0
            count = 0
            for batch in dev_iter:
                for i in batch.index:
                    assert len(counter_dev[i].sentence) <= batch.sentence.shape[0], "graph out-side"

                x = batch.sentence.t().to(device)
                adj = []
                trigger = batch.trigger_index.t().flatten().to(device)

                for ind in batch.index:
                    adj.append(counter_dev[ind].trans_data(x.shape[-1]))
                adj = torch.stack(adj, 0).to(device).to_dense()

                mask = get_pad_mask(x, TEXT.vocab.stoi[TEXT.pad_token]).to(device)
                eep = batch.eep.squeeze().to(device)
                # print(x.shape)
                # print(adj.shape)
                # print(trigger.shape)
                # print(mask.shape)

                out = model(x, trigger, adj)

                loss = F.l1_loss(out, eep)
                temp_loss += loss.item()
                count += 1
            # print("dev loss:",(temp_loss / count))
            loss_list += (temp_loss / count)
            return (loss_list / epoch)

    def trainer_test(epoch):
        loss_list = 0.0
        eval_history_out = []
        eval_history_label = []
        for cycle in trange(epoch):
            model.eval()
            temp_loss = 0.0
            count = 0
            for batch in test_iter:
                for i in batch.index:
                    assert len(counter_test[i].sentence) <= batch.sentence.shape[0], "graph out-side"

                x = batch.sentence.t().to(device)
                adj = []
                trigger = batch.trigger_index.t().flatten().to(device)
                if x.shape[0] == 1:
                    continue

                for ind in batch.index:
                    adj.append(counter_test[ind].trans_data(x.shape[-1]))
                adj = torch.stack(adj, 0).to(device).to_dense()

                mask = get_pad_mask(x, TEXT.vocab.stoi[TEXT.pad_token]).to(device)
                eep = batch.eep.squeeze().to(device)
                # print(x.shape)
                # print(adj.shape)
                # print(trigger.shape)
                # print(mask.shape)

                out = model(x, trigger, adj)

                loss = F.l1_loss(out, eep)
                temp_loss += loss.item()
                writer.add_pr_curve('pr_curve', out, eep, 0)
                count += 1
                eval_history_out = eval_history_out + out.cpu().detach().numpy().tolist()
                eval_history_label = eval_history_label + eep.cpu().detach().numpy().tolist()

            loss_list += (temp_loss / count)
            print("test", loss_list)
            r = pearsonr(eval_history_out, eval_history_label)
            print(r)
            return (loss_list / epoch)

    if resume:  # resume为参数，第一次训练时设为0，中断再训练时设为1

        model_path = os.path.join('/media/user1/325655435655094D/baseline/checkpoint/uds',
                                  'smodel_bert_new.pth.tar')
        assert os.path.isfile(model_path)
        # model.load_state_dict(torch.load(model_path))
        checkpoint = torch.load(model_path)
        best_acc = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Load checkpoint at epoch {}.'.format(start_epoch))
        print('Best accuracy so far {}.'.format(best_acc))

    trainer_train(epoch)


if __name__ == '__main__':

    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_parmas())
        params.update(tuner_params)
        print(params)
        run(params)
    except Exception as exception:
        logger.exception(exception)
        raise
