import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cola import COLA
from models.similarity import BilinearSimilarity
from data.audioset import Audioset
from data.utils import collate_audio_data


def main(args):
    cola = COLA(args.hidden_size, args.output_size)
    similarity = BilinearSimilarity(args.output_size)
    cross_entropy = nn.CrossEntropyLoss()
    
    audioset = Audioset(args.audioset_csv)
    dataloader = DataLoader(audioset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_audio_data)
    optimizer = optim.AdamW(cola.parameters(), lr=args.learning_rate)
    with open('test_files.txt', 'r') as f:
        audio_names = [line.strip() for line in f]
    anchors = torch.load('test_anchors.pth')
    positives = torch.load('test_positives.pth')
    batch = audio_names, anchors, positives

    cola.cuda()
    similarity.cuda()
    for ix, batch in enumerate(dataloader):
        audio_names, anchors, positives = batch
        n_batch = anchors.shape[0]
        anchors = anchors.cuda()
        positives = positives.cuda()
        optimizer.zero_grad()

        y_anchors = cola(anchors)
        y_positives = cola(positives)

        similarities = similarity(y_anchors, y_positives)
        loss = cross_entropy(similarities, torch.arange(n_batch).cuda())

        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()} :)')
    # TODO: dane, configi, wszystko
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train COLA in PyTorch')
    # TODO: arguments
    parser.add_argument('--hidden_size', help='Size of the tensor after encoder', default=1280)
    parser.add_argument('--output_size', help='Size of the tensor after the projection head', default=512)
    parser.add_argument('--learning_rate', help='Learning rate', default=0.0001)
    parser.add_argument('--batch_size', help='Batch size', default=64)
    parser.add_argument('--no_of_epochs', help='Number of epochs', default=500)
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('audioset_csv', help='Path to the Audioset unbalanced train set')
    args = parser.parse_args()
    main(args)
