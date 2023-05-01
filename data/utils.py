import torch


def collate_audio_data(samples):
    filtered_batch = [sample for sample in samples if sample[0] is not None]
    audio_names, anchors, positives = zip(*filtered_batch)

    anchors = torch.stack([torch.tensor(x) for x in anchors])
    positives = torch.stack([torch.tensor(x) for x in positives])

    return audio_names, anchors, positives
