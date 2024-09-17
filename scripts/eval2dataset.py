# convert an evaluation file to a PSG dataset for two-stage methods
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
import pickle
import json
import numpy as np

MAX_SEG_ID = 256**3 - 1


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        assert id_map.max() <= 256 * 256 * 256 - 1, id_map.max()
        rgb_map = np.empty((*id_map.shape, 3), dtype=np.uint8)
        rgb_map[..., 0] = id_map % 256
        rgb_map[..., 1] = (id_map // 256) % 256
        rgb_map[..., 2] = (id_map // 256 // 256) % 256
        return rgb_map
    color = [id_map % 256, (id_map // 256) % 256, (id_map // 256 // 256) % 256]
    return color


def cli():
    parser = ArgumentParser()
    parser.add_argument("eval")
    parser.add_argument("psg")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    assert not out_dir.exists()

    with open(args.eval, "rb") as f:
        eval_data = pickle.load(f)
    with open(args.psg) as f:
        psg = json.load(f)

    byid = {}
    for x in psg["data"]:
        byid[x["image_id"]] = x

    out_dir.mkdir(exist_ok=False, parents=True)

    new_data = []
    for r in eval_data:
        orig = byid[r["img_id"]]
        annotations = []
        segments_info = []
        seg_ids = np.linspace(0, MAX_SEG_ID, num=len(r["box_label"]) + 1, dtype=int)
        seg_mask = np.zeros_like(r["mask"])
        for i, (lbl, si) in enumerate(zip(r["box_label"].tolist(), seg_ids[1:])):
            m = r["mask"] == i
            seg_mask[m] = si
            y, x = np.where(m)
            annotations.append(
                {
                    "bbox": [
                        int(np.min(x)),
                        int(np.min(y)),
                        int(np.max(x)),
                        int(np.max(y)),
                    ],
                    "bbox_mode": 0,
                    "category_id": lbl,
                }
            )
            segments_info.append(
                {
                    "id": int(si),
                    "category_id": lbl,
                    "iscrows": 0,
                    "isthing": 1 if lbl < 80 else 0,
                    "area": int(m.sum()),
                }
            )
        new_data.append(
            {
                "image_id": r["img_id"],
                "coco_image_id": orig["coco_image_id"],
                "height": orig["height"],
                "width": orig["width"],
                "file_name": orig["file_name"],
                "pan_seg_file_name": orig["pan_seg_file_name"],
                "annotations": annotations,
                "segments_info": segments_info,
            }
        )

        # write PNG image
        # convert ids to seg ids, then convert to PNG and save
        img = Image.fromarray(id2rgb(seg_mask))
        out_p = out_dir / orig["pan_seg_file_name"]
        out_p.parent.mkdir(exist_ok=True, parents=True)
        img.save(out_p)

    with open(out_dir / "psg.json", "w") as f:
        json.dump(
            {
                "data": new_data,
                "thing_classes": psg["thing_classes"],
                "stuff_classes": psg["stuff_classes"],
                "predicate_classes": psg["predicate_classes"],
                "test_image_ids": psg["test_image_ids"],
            },
            f,
        )


if __name__ == "__main__":
    cli()
