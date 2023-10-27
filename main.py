import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from models.cola import COLA
from models.similarity import BilinearSimilarity
from data.audioset import Audioset, LocalAudioset
from data.utils import collate_audio_data


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cola = COLA(args.hidden_size, args.output_size)
    similarity = BilinearSimilarity(args.output_size)
    cross_entropy = nn.CrossEntropyLoss()

    # audioset_train = Audioset(args.audioset_csv, quiet=True)
    audioset_train = LocalAudioset(args.audioset_train_folder, args.audioset_train_paths)
    audioset_val = LocalAudioset(args.audioset_valid_folder, args.audioset_valid_paths)
    dataloader_train = DataLoader(audioset_train, batch_size=args.batch_size, num_workers=8, collate_fn=collate_audio_data)
    dataloader_val = DataLoader(audioset_val, batch_size=args.batch_size, num_workers=4, collate_fn=collate_audio_data)
    if not args.finetune:
        optimizer = optim.AdamW(cola.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam([
            {'params': cola.encoder.backend.features.parameters(), 'lr': args.learning_rate / 5},
            {'params': cola.encoder.fc.parameters(), 'lr': args.learning_rate / 2},
            {'params': cola.projection.parameters()},
            {'params': cola.layernorm.parameters()},
        ], lr=args.learning_rate)
    # with open('test_files.txt', 'r') as f:
    #     audio_names = [line.strip() for line in f]
    # anchors = torch.load('test_anchors.pth')
    # positives = torch.load('test_positives.pth')
    # batch = audio_names, anchors, positives

    if args.log:
        run = wandb.init(project='COLA-PyTorch')
        run.config.learning_rate = args.learning_rate
        run.config.batch_size = args.batch_size
        run.config.hidden_size = args.hidden_size
        run.config.output_size = args.output_size

    cola.cuda()
    similarity.cuda()
    for epoch in range(args.no_of_epochs):
        # Training loop
        for ix, batch in tqdm(enumerate(dataloader_train)):
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
            print(f'Epoch: {epoch}, Batch: {ix}, Training Loss: {loss.item()}')
            run.log({'loss': loss.item()})
        if ix % args.save_every == 0:
            torch.save(cola.state_dict(), os.path.join(args.output_dir, f'cola_{ix}.pth'))
            torch.save(similarity.state_dict(), os.path.join(args.output_dir, f'similarity_{ix}.pth'))
        with torch.no_grad():
            val_loss = 0
            for ix, batch in tqdm(enumerate(dataloader_val)):
                audio_names, anchors, positives = batch
                n_batch = anchors.shape[0]
                anchors = anchors.cuda()
                positives = positives.cuda()

                y_anchors = cola(anchors)
                y_positives = cola(positives)

                similarities = similarity(y_anchors, y_positives)
                loss = cross_entropy(similarities, torch.arange(n_batch).cuda())
                val_loss += loss.item()
            val_loss /= len(dataloader_val)
            print(f'Epoch: {epoch}, Validation Loss: {val_loss}')
            run.log({'valid_loss': val_loss})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train COLA in PyTorch')
    # TODO: arguments
    parser.add_argument('--hidden_size', help='Size of the tensor after encoder', type=int, default=1280)
    parser.add_argument('--output_size', help='Size of the tensor after the projection head', type=int, default=512)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=64)
    parser.add_argument('--no_of_epochs', help='Number of epochs', type=int, default=500)
    parser.add_argument('--log', help='Log using wandb', type=bool, default=True)
    parser.add_argument('--save_every', help='Save checkpoint every n epochs', type=int, default=1)
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--audioset_csv', help='Path to the Audioset unbalanced train set')
    parser.add_argument('--audioset_train_folder', help='Path to the Audioset balanced train set')
    parser.add_argument('--audioset_train_paths', help='Text file where lines are the train audio paths', required=False, default=None)
    parser.add_argument('--audioset_valid_folder', help='Path to the Audioset balanced valid set')
    parser.add_argument('--audioset_valid_paths', help='Text file where lines are the valid audio paths', required=False, default=None)
    parser.add_argument('--finetune', action='store_true', help='Finetune layers after the encoder')
    args = parser.parse_args()
    print(args)
    main(args)
