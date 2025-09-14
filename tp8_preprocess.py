import array
import csv
import gzip
import logging
import re
import shutil
import sys
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm
import kagglehub
import click
import sentencepiece as spm
import torch

logging.basicConfig(level=logging.INFO)

MAINDIR = Path("./")
DATA_PATH = MAINDIR.joinpath("data")
DATA_PATH.mkdir(exist_ok=True)
SRC_PATH = DATA_PATH.joinpath("training.1600000.processed.noemoticon.csv")
RE_URL = re.compile(r"(?:\@|https?\://)\S+")
RE_MENTION = re.compile(r"(?:@)\S+")
RE_NOT = re.compile(r"[^\w\s@:,;]+")

# Téléchargement des données depuis Kaggle
def download_dataset():
    if SRC_PATH.is_file():
        logging.info(f"Dataset déjà présent : {SRC_PATH}")
        return SRC_PATH
    
    logging.info("Téléchargement du dataset Sentiment140 depuis Kaggle...")
    try:
        kaggle_path = kagglehub.dataset_download("kazanova/sentiment140")
        csv_file = list(Path(kaggle_path).glob("*.csv"))[0]
        shutil.copy(str(csv_file), str(SRC_PATH))
        logging.info(f"Dataset téléchargé et copié vers : {SRC_PATH}")
        return SRC_PATH
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement : {e}")
        raise RuntimeError("Impossible de télécharger le dataset depuis Kaggle.")

# Nettoyage des données
def datareader(path: Path):
    with open(path, "rt", encoding="utf-8", errors="ignore") as fp:
        for row in csv.reader(fp):
            if len(row) >= 6:  # Vérifie que la ligne contient suffisamment de colonnes
                tweet = RE_NOT.sub(" ", RE_MENTION.sub("@USER", RE_URL.sub("", row[5])))
                yield tweet.strip(), row[0]

# Création du fichier intermédiaire nettoyé
def cleanup(src, target):
    """Nettoyage du jeu de tweets et création d'un fichier intermédiaire."""
    if not target.is_file():
        logging.info(f"Création du fichier texte nettoyé depuis {src}")
        target_tmp = target.with_suffix(".tmp")
        with target_tmp.open("wt", encoding="utf-8") as out:
            for tweet, _ in tqdm(datareader(src), desc="Nettoyage des tweets"):
                out.write(tweet)
                out.write("\n")
        shutil.move(target_tmp, target)

# Dataset PyTorch
Batch = namedtuple("Batch", ["text", "labels"])

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text: torch.LongTensor, sizes: torch.LongTensor, labels: torch.LongTensor):
        self.text = text
        self.sizes = sizes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[self.sizes[index]:self.sizes[index+1]], self.labels[index].item()

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), torch.LongTensor(labels))

# Création des données tokenisées
def generatedata(mode: str, tokenizer, vocab_size: int, fn, map):
    datapath = MAINDIR / f"{mode}-{vocab_size}.pth"
    if datapath.is_file():
        logging.info(f"Le fichier {datapath} existe déjà.")
        return

    text = array.array("L")
    sizes = array.array("L")
    labels = array.array("B")
    sizes.append(0)

    count = 0
    for tweet, label in tqdm(datareader(fn), desc=f"Tokenisation des tweets {mode}"):
        if int(label) in map:
            tokens = tokenizer.encode_as_ids(tweet)
            for tokenid in tokens:
                text.append(tokenid)
            sizes.append(len(text))
            labels.append(map[int(label)])
            count += 1

    logging.info(f"Processed {count} tweets for {mode}")
    data = TextDataset(torch.LongTensor(text), torch.LongTensor(sizes), torch.LongTensor(labels))
    with gzip.open(datapath, "wb") as fp:
        torch.save(data, fp)
    logging.info(f"Saved {mode} dataset with {len(labels)} examples")

# CLI principale
@click.option("--vocab-size", default=1000, type=int, help="Taille du vocabulaire")
@click.command()
def cli(vocab_size: int):
    # Téléchargement des données
    download_dataset()

    # Nettoyage et création du fichier intermédiaire
    TRAINPATH = DATA_PATH.joinpath("sentiment140-train.txt")
    cleanup(SRC_PATH, TRAINPATH)

    # Création du vocabulaire SentencePiece
    wpmodel = Path(f"wp{vocab_size}.model")
    if not wpmodel.is_file():
        logging.info(f"Création du modèle SentencePiece avec vocabulaire {vocab_size}")
        spm.SentencePieceTrainer.train(
            input=str(TRAINPATH),
            model_prefix=f"wp{vocab_size}",
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="unigram",
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
        )

    # Chargement du tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f"wp{vocab_size}.model")

    # Tokenisation et sauvegarde des données
    CLASSMAP = {0: 0, 4: 1}  # Classe négative : 0, Classe positive : 1
    generatedata("train", tokenizer, vocab_size, SRC_PATH, CLASSMAP)

if __name__ == "__main__":
    cli()
