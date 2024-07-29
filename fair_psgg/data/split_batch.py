import torch


# split a batch into smaller batches if there are too many sampled relations
def split_batch(batch: dict, max_relations: int):
    """Splits a batch into smaller batches if there are too many relations per image.
    `num_relations` is changed accordingly.
    """
    relations = batch["sampled_relations"]

    total_num_rel = len(relations)
    num_blocks = total_num_rel // max_relations
    if total_num_rel % max_relations > 0:
        num_blocks += 1
    rel_block_size = round(total_num_rel / num_blocks)
    block_sizes = torch.tensor(
        [len(t) for t in torch.arange(total_num_rel).split(rel_block_size)]
    )
    rel_block_ids = torch.repeat_interleave(block_sizes)
    img_id_for_rel = torch.repeat_interleave(batch["num_relations"])
    img_id_for_box = torch.repeat_interleave(batch["num_boxes"])

    for block_id in range(len(block_sizes)):
        # select all relations that should land in this sub batch
        sub_rel_mask = rel_block_ids == block_id

        # get the local ids of the selected images
        # local means inside the batch (therefore id is from 0 to batch_size-1)
        sub_img_ids, sub_num_relations = img_id_for_rel[sub_rel_mask].unique(
            return_counts=True
        )
        # select all boxes that belong to the selected subset of images
        sub_box_mask = (img_id_for_box[:, None] == sub_img_ids[None]).any(dim=1)

        sub_batch = {
            "image_id": batch["image_id"][sub_img_ids],
            "img": batch["img"][sub_img_ids],
            "num_boxes": batch["num_boxes"][sub_img_ids],
            "bboxes": batch["bboxes"][sub_box_mask],
            "box_categories": batch["box_categories"][sub_box_mask],
            "num_relations": sub_num_relations,
            "sampled_relations": relations[sub_rel_mask],
        }
        if "segmentation" in batch:
            sub_batch["segmentation"] = batch["segmentation"][sub_box_mask]
        if "idx" in batch:
            sub_batch["idx"] = batch["idx"][sub_img_ids]
        assert sub_batch["num_relations"].shape == sub_batch["num_boxes"].shape
        yield sub_batch


def split_batch_iter(batches, max_relations: int):
    """Iterator version of `split_batch`.
    Splits a batch into smaller batches if there are too many relations per image.
    `num_relations` is changed accordingly.
    """
    for batch in batches:
        for sub in split_batch(batch, max_relations):
            yield sub
