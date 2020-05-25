import os, torch, torchaudio, librosa
from torch.utils import data

fsdd_partitions = {'train': [], 'validation': [], 'test': []}
fsdd_labels = {}

for file_name in os.listdir('../recordings'):
    file_name = file_name[:-4]
    digit, name, iteration = file_name.split('_')
    digit, iteration = int(digit), int(iteration)

    # Partition files into 80-10-10 training, validation and test set split
    if iteration < 10:
        fsdd_partitions['validation'].append(file_name)
    else:
        if iteration <= 20:
            fsdd_partitions['test'].append(file_name)
        else:
            fsdd_partitions['train'].append(file_name)

    # Store the label for this file
    fsdd_labels[file_name] = digit

class FSDD:
    def __init__(self, partition):
        self.partition = fsdd_partitions[partition]
        self.labels = fsdd_labels

    def load(self):
        XX, yy = [], []

        for id_ in self.partition:
            X, _ = librosa.load(f'../recordings/{id_}.wav', mono=True)
            XX.append(X)
            yy.append(str(self.labels[id_]))

        return XX, yy

class TorchFSDD(data.Dataset):
    def __init__(self, partition, transform=None):
        self.partition = fsdd_partitions[partition]
        self.labels = fsdd_labels
        self.transform = transform

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, index):
        # Select sample(s)
        id_ = self.partition[index]

        # Load data and get label
        X, _ = torchaudio.load(f'../recordings/{id_}.wav')
        y = self.labels[id_]

        # Transform the data if a transformation is provided
        if self.transform is not None:
            X = self.transform(X)

        return X, y

def collate_fn(batch):
    """Collation function from https://www.codefull.org/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/.

    Parameters
    ----------
    batch: list(tuple(Tensor, int))
        Collection of (sequence, label) pairs, where the nth sequence is of shape 1xDxTn
    """

    # Sort the (mfccs, label) pairs in descending order of recording duration
    batch.sort(key=(lambda x: x[0].shape[-1]), reverse=True)
    # Shape: list(tuple(1xDxT, int))

    # Get each MFCC sequence, remove first dimension, transpose it, and pad it
    sequences = [mfccs.squeeze(0).permute(1, 0) for mfccs, _ in batch]
    # Shape: list(TxD)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Shape: BxTxD

    # Store the lengths of each unpadded sequence (in order to unpad later)
    lengths = torch.LongTensor([len(mfccs) for mfccs in sequences])
    # Shape: B

    # Retrieve the labels of the sorted batch
    labels = torch.LongTensor([label for _, label in batch])
    # Shape: B

    return padded_sequences, lengths, labels
    # Shapes: BxTxD, B, B