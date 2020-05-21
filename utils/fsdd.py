import os, torch, torchaudio, librosa
from torch.utils import data

partitions = {'train': [], 'validation': [], 'test': []}
labels = {}

for file_name in os.listdir('../recordings'):
    file_name = file_name[:-4]
    digit, name, iteration = file_name.split('_')
    digit, iteration = int(digit), int(iteration)

    # Partition files into 80-10-10 training, validation and test set split
    if iteration < 10:
        partitions['validation'].append(file_name)
    else:
        if iteration <= 20:
            partitions['test'].append(file_name)
        else:
            partitions['train'].append(file_name)

    # Store the label for this file
    labels[file_name] = digit

class FSDD:
    def __init__(self, partition, labels=labels):
        self.partition = partitions[partition]
        self.labels = labels
        self.sample_rate = 8000

    def load(self):
        XX, yy = [], []

        for id_ in self.partition:
            X, _ = librosa.load(f'../recordings/{id_}.wav', mono=True)
            XX.append(X)
            yy.append(str(self.labels[id_]))

        return XX, yy

class TorchFSDD(data.Dataset):
    def __init__(self, partition, labels=labels):
        self.partition = partitions[partition]
        self.labels = labels
        self.sample_rate = 8000

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, index):
        # Select sample
        id_ = self.partition[index]

        # Load data and get label
        X, _ = torchaudio.load(f'../recordings/{id_}.wav')
        y = self.labels[id_]

        return X, y