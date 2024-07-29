# functions used by different datasets/dataloaders
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader


def custom_collate(samples):
    collated = defaultdict(list)
    for sample in samples:
        for k, v in sample.items():
            # only tensors can be stacked/concatenated
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            collated[k].append(v)

    output = {}
    for key in collated:
        if key in ("img", "idx", "image_id"):
            output[key] = torch.stack(collated[key])
        else:
            output[key] = torch.cat(collated[key])
            if key == "bboxes":
                output["num_boxes"] = torch.tensor([len(x) for x in collated[key]])

            if key == "sampled_relations":
                output["num_relations"] = torch.tensor([len(x) for x in collated[key]])

    assert output["num_boxes"].shape == output["num_relations"].shape

    return output


def get_generic_loader(
    dataset: Dataset, batch_size: int, num_workers: int, is_train: bool
):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
    )
