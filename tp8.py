import logging
from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)
import heapq
from pathlib import Path
import gzip
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
from tp8_preprocess import TextDataset
import torch.optim as optim

# Configuration
vocab_size = 1000
MAINDIR = Path(__file__).parent
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)

train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500

# Chargement des jeux de données
val_size = 1000
test_size = 10000
train_size = len(train) - val_size - test_size
train, val, test = torch.utils.data.random_split(train, [train_size, val_size, test_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)

class BaselineModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.majority_class = 0
        
    def forward(self, x):
        return torch.full((x.shape[0], 2), float('-inf'), device=x.device).scatter_(
            1, torch.full((x.shape[0], 1), self.majority_class, device=x.device).long(), 0
        )

class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

class DeeperCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

class WideKernelCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

def evaluate(model, data_iter, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in data_iter:
            texts, labels = batch.text.to(device), batch.labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(data_iter), 100. * correct / total

def train_model(model, train_iter, val_iter, epochs=5, device="cuda"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter()
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_iter, desc=f"Epoch {epoch+1}/{epochs}"):
            texts, labels = batch.text.to(device), batch.labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = 100. * correct / total
        val_loss, val_acc = evaluate(model, val_iter, criterion, device)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model.__class__.__name__}_best.pth')
    writer.close()
    return model, best_val_acc

def extract_activations(model, data_iter, tokenizer, device):
    model.eval()
    W_init = 1
    S_init = 1
    
    layers = [
        (model.conv1.kernel_size[0], model.conv1.stride[0]),
        (model.pool1.kernel_size, model.pool1.stride),
        (model.conv2.kernel_size[0], model.conv2.stride[0]),
        (model.pool2.kernel_size, model.pool2.stride)
    ]
    
    W, S = W_init, S_init
    for w_i, s_i in layers:
        W = W + (w_i - 1) * S
        S = S * s_i
    
    activations = {i: {"max_activation": float('-inf'), "sequence": None, 
                      "full_text": None, "indices": None} 
                  for i in range(model.conv2.out_channels)}
    
    with torch.no_grad():
        for batch in tqdm(data_iter, desc="Analyzing activations"):
            texts = batch.text.to(device)
            embeddings = model.embedding(texts).transpose(1, 2)
            x = torch.relu(model.conv1(embeddings))
            x = model.pool1(x)
            x = torch.relu(model.conv2(x))
            
            for filter_idx in range(x.size(1)):
                for seq_idx in range(x.size(0)):
                    activations_seq = x[seq_idx, filter_idx]
                    max_activation, max_idx = activations_seq.max(dim=0)
                    
                    if max_activation > activations[filter_idx]["max_activation"]:
                        start = max_idx.item() * S
                        end = start + W
                        full_text = tokenizer.DecodeIds(texts[seq_idx].cpu().tolist())
                        words = full_text.split()
                        if end <= len(words):
                            subsequence = " ".join(words[start:end])
                            activations[filter_idx] = {
                                "max_activation": max_activation.item(),
                                "sequence": subsequence,
                                "full_text": full_text,
                                "indices": (start, end)
                            }
    return activations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Baseline
    all_labels = []
    for batch in train_iter:
        all_labels.extend(batch.labels.tolist())
    majority_class = max(set(all_labels), key=all_labels.count)
    baseline = BaselineModel()
    baseline.majority_class = majority_class
    baseline = baseline.to(device)
    _, baseline_acc = evaluate(baseline, test_iter, nn.CrossEntropyLoss(), device)
    print(f"Baseline Test Accuracy: {baseline_acc:.2f}%")

    # Entraînement des modèles
    models = {
        "SimpleCNN": SimpleCNN(ntokens),
        "DeeperCNN": DeeperCNN(ntokens),
        "WideKernelCNN": WideKernelCNN(ntokens),
    }

    # Entraînement et évaluation
    for name, model in models.items():
        print(f"\nTraining {name}")
        trained_model, val_acc = train_model(model, train_iter, val_iter, device=device)
        _, test_acc = evaluate(trained_model, test_iter, nn.CrossEntropyLoss(), device)
        print(f"{name} - Validation Accuracy: {val_acc:.2f}% | Test Accuracy: {test_acc:.2f}%")
        
        print(f"\nAnalyse des activations pour {name}")
        model.load_state_dict(torch.load(f'{name}_best.pth'))
        model = model.to(device)
        
        print("\nRecherche des sous-séquences les plus activantes...")
        activations = extract_activations(model, train_iter, tokenizer, device)
        
        print(f"\nSous-séquences activant le plus les filtres de {name}:")
        for filter_idx, data in activations.items():
            print(f"\nFiltre {filter_idx}:")
            print(f"  Activation: {data['max_activation']:.4f}")
            print(f"  Sous-séquence: {data['sequence']}")
            print(f"  Indices: {data['indices']}")
            print(f"  Texte complet: {data['full_text']}")

if __name__ == "__main__":
    main()