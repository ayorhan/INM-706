# dataset_loader.py
import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Hyperparameters
BATCH_SIZE = 64
MAX_SEQ_LEN = 256
VOCAB_SIZE = 5000

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Special tokens
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

print("Loading dataset...")

# Load the dataset
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

print("Building vocabulary...")

# Function to yield tokens from the dataset
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Build the vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), max_tokens=VOCAB_SIZE)
vocab.insert_token(UNK_TOKEN, 0)
vocab.insert_token(PAD_TOKEN, 1)
vocab.set_default_index(vocab[UNK_TOKEN])  # Set unknown tokens to default index

print(f"Vocabulary size: {len(vocab)}")

# Function to preprocess the text
def preprocess(text):
    return vocab(tokenizer(text))[:MAX_SEQ_LEN]

# Collate function for DataLoader
def collate_batch(batch):
    labels, texts = zip(*batch)
    labels = torch.tensor([1 if label == "pos" else 0 for label in labels], dtype=torch.int64)
    texts = [torch.tensor(preprocess(text), dtype=torch.int64) for text in texts]
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab[PAD_TOKEN])
    return labels, texts

print("Creating data loaders...")

# Create DataLoaders
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')
train_loader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, collate_fn=collate_batch)

print("Data loaders created successfully.")

# Test Code
def test_data_loader():
    for labels, texts in train_loader:
        print("Labels batch shape:", labels.size())
        print("Texts batch shape:", texts.size())
        print("Example text:", texts[0])
        print("Example label:", labels[0])
        break

if __name__ == "__main__":
    print("Testing data loader...")
    test_data_loader()
    print("Data loader test completed successfully.")
