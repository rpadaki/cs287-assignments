import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

# Fields for processing  
TEXT = NamedField(names=('seqlen',))
LABEL = NamedField(sequential=False, names=(), unk_token=None)

# Split data into train, validation, test 
train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

# Build vocab
TEXT.build_vocab(train, vectors='glove.6B.100d')
LABEL.build_vocab(train)

# Set up batches for model input  
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
  (train, val, test), batch_size=128, device=torch.device('cuda'))


class CNN(nn.Module):
    def __init__(
        self,
        num_classes=2,
        kernel_sizes=[2, 3, 4],
        num_filters=100,
        vocab_size=16284,
        embedding_dim=300,
        embedding2_dim=100,
        dropout=0.5,
        pretrained_embeddings=None,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings[0])
        self.embedding2 = nn.Embedding(vocab_size, embedding2_dim)
        self.embedding2.weight.data.copy_(pretrained_embeddings[1])
        
        self.embedding.weight.requires_grad = False
        self.embedding2.weight.requires_grad = False
        

        conv_blocks = []
        for kernel_size in kernel_sizes:
            conv1d = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
            )
            conv_blocks.append(conv1d)
        for kernel_size in kernel_sizes:
            conv1d = nn.Conv1d(
                in_channels=embedding2_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
            )
            conv_blocks.append(conv1d)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nn.Linear(num_filters * len(kernel_sizes) * 2, num_classes)


class NamedCNN(CNN):
    def forward(self, x):  # x: (batch, seqlen)
        x1 = x.augment(self.embedding, "h") \
             .transpose("h", "seqlen")

        x1_list = [x1.op(conv_block, F.relu).max("seqlen")[0]
                   for conv_block in self.conv_blocks[:3]]
        x2 = x.augment(self.embedding2, "h") \
             .transpose("h", "seqlen")
        x2_list = [x2.op(conv_block, F.relu).max("seqlen")[0]
                   for conv_block in self.conv_blocks[3:]]
        x1_list.extend(x2_list)
        out = ntorch.cat(x1_list, "h")
#         print(out.shape)

        feature_extracted = out
        drop = lambda x: F.dropout(x, p=0.5, training=self.training)
        out = out.op(drop, self.fc, classes="h").softmax("classes")
#         print(out.shape)

        return out, feature_extracted


num_classes = 2
num_filters = 100
kernel_sizes = [3, 4, 5]
stride_lengths = [1]
batch_size = 128
dropout = 0.5

pretrained_embeddings = [TEXT.vocab.vectors, TEXT_WIKI.vocab.vectors]
print(pretrained_embeddings[0].size())
print(pretrained_embeddings[1].size())
vocab_size = pretrained_embeddings[0].size()[0]
embedding_dim = pretrained_embeddings[0].size()[1]
embedding2_dim = pretrained_embeddings[1].size()[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NamedCNN(num_classes, kernel_sizes, num_filters, vocab_size, 
            embedding_dim, embedding2_dim, dropout, pretrained_embeddings)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.0002)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


# Accuracy helper calculation
def batch_binary_accuracy(preds, labels):
    """Return accuracy per batch"""
    return (preds == labels).sum("batch").item() / len(labels)


def train(model, train_iter, criterion, optimizer):
    """Training Function"""
    model.train()

    epoch_loss = 0
    loss_fn = criterion

    for batch in train_iter:
        optimizer.zero_grad()
        preds, vector = model(batch.text)
#         print(ntorch.tensor(batch.label).unsqueeze(1).shape)
        loss = preds.reduce2(batch.label, loss_fn, ("batch", "classes"))
#         loss = preds.reduce2(batch.label, loss_fn, ("batch"))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
      
    return epoch_loss / len(train_iter)


def validate(model, val_iter, criterion):
    """Validation function, called at the end of each epoch"""
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in val_iter:
            preds, vector = model(batch.text)
            preds = preds.max("classes")[1]
            acc = batch_binary_accuracy(preds, batch.label)
            
            epoch_acc += acc
            
    return epoch_acc / len(val_iter)


N_EPOCHS = 30

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iter, criterion, optimizer)
    val_acc = validate(model, val_iter, criterion)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | \
          Val. Acc: {val_acc*100:.2f}% |')