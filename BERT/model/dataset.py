from torch.utils.data import DataLoader,Dataset

class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        data=self.dataset[index]
        return data

def construct(filename):
    f=open(filename,encoding='utf8')
    list=[]
    for line in f:
        line=line.replace("\n","")
        pairs=line.split(" ")
        elem={'input':pairs[0],'output':pairs[1]}
        list.append(elem)
    return list

def singleconstruct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        if (len(pairs[0]) != len(pairs[1])):
            continue
        elem = {'input': pairs[1], 'output': pairs[1]}
        list.append(elem)
    return list