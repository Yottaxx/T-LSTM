loaddata = LoadData("./unified/uw/train.conll", "./unified/uw/dev.conll", "./unified/uw/test.conll")

counter, counter_dev, counter_test = [],[],[]

a = loaddata.conllu_counter['train']
a = loaddata.counter_process(a)
for d in a:
    counter.append(d)
for d in range(len(counter)):
    counter[d].index=tuple([d])
b = loaddata.conllu_counter['dev']
b = loaddata.counter_process(b)
for d in b :
    counter_dev.append(d)
for d in range(len(counter_dev)):
    counter_dev[d].index=tuple([d])
c = loaddata.conllu_counter['test']
c = loaddata.counter_process(c)
test_i=c.copy()
for d in c:
    counter_test.append(d)
for d in range(len(counter_test)):
    counter_test[d].index=tuple([d])