import itertools
import logging
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse_incr
logging.basicConfig(level=logging.INFO)

DATA_PATH = "./mes_projets/AMAL/student_tp5/data/"

# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, oov_rate=0.1, adding=True):
        self.sentences = []
        for s in data:
            sentence_tokens = [words.get(token["form"], adding) for token in s]
            sentence_tags = [tags.get(token["upostag"], adding) for token in s]
            if adding and torch.rand(1).item() < oov_rate:
                rand_index = torch.randint(0, len(sentence_tokens), (1,)).item()
                sentence_tokens[rand_index] = Vocabulary.OOVID
            self.sentences.append((sentence_tokens, sentence_tags))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]

def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu", "r", encoding="utf-8")
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)
data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu", "r", encoding="utf-8")
dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)
data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu","r", encoding="utf-8")
test_data = TaggingDataset(parse_incr(data_file), words, tags, False)

logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=32

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)



#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)

class Tagger(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_tags):
        super(Tagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=Vocabulary.PAD)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, num_tags)  

    def forward(self, x, lengths):
        x = self.embedding(x)  # Embedding layer
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)
        final_output = self.linear(output_padded)
        return final_output

def evaluate(model, dev_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in tqdm(dev_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            lengths = torch.tensor([len(seq[seq != Vocabulary.PAD]) for seq in x.T])
            output = model(x, lengths)
            output = output.view(-1, output.shape[2])
            y = y.view(-1)
            loss = loss_fn(output, y)
            total_loss += loss.item()
        
    return total_loss / len(dev_loader)

def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        lengths = torch.tensor([len(seq[seq != Vocabulary.PAD]) for seq in x.T])
        optimizer.zero_grad()
        output = model(x, lengths)
        output = output.view(-1, output.shape[2])
        y = y.view(-1)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def fit(model, train_loader, dev_loader, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    device = "cpu"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)

    try:
        current_epoch = torch.load("temp_epoch")+1
        epochs = epochs-current_epoch+1
        epochs = epochs if epochs >= 0 else 0 
        model.load_state_dict(torch.load(f"temp_model_{current_epoch}"))
        optimizer.load_state_dict(torch.load(f"temp_opt_{current_epoch}"))
        print(f"[INFO] Loaded state at epoch {current_epoch}")
            
    except Exception as e:
        print(f"[ERROR] Nothing to load")
        pass

    train_losses = []
    val_losses = []
    for e in range(epochs):
        train_loss = train(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate(model, dev_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {e} train loss: {train_loss} val loss: {val_loss}")
        torch.save(model.state_dict(),f"temp_model_{e}")
        torch.save(optimizer.state_dict(), f"temp_opt_{e}")
        torch.save(e,"temp_epoch")
    return train_losses, val_losses


def predict(model, sentence, vocab, tag_vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        tokens = [vocab.get(token, adding=False) for token in sentence.split(' ')]
        lengths = torch.tensor([len(tokens)])
        tokens = torch.tensor(tokens).unsqueeze(0).to(device) 
        tokens = tokens.permute(1, 0) 
        outputs = model(tokens, lengths)
        _, predicted_tags = torch.max(outputs, dim=-1)
        
    predicted_tags = [tag_vocab.getword(idx) for idx in predicted_tags[:, 0].cpu().numpy()]
    print(f"Input: {sentence}")
    print(f"Prediction: {predicted_tags}")


def plot_compare(train_loss, val_loss):
    plt.style.use("ggplot")
    plt.figure(figsize=(12,7))
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Val loss")
    plt.legend()
    plt.title("Train loss and Validation loss")
    plt.show()

if __name__ == "__main__":
    LATENT_SPACE = 128
    VOCAB_SIZE = len(words)
    TAG_SIZE = len(tags)
    model = Tagger(LATENT_SPACE, VOCAB_SIZE, TAG_SIZE)
    train_losses, val_losses = fit(model, train_loader, dev_loader, epochs=10, lr=1e-3)
    sentence = "Salut , je suis Akli, voici mon projet"
    predicted_tags = predict(model, sentence, words, tags)