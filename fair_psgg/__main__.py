from argparse import ArgumentParser
import json
from project_paths import project_paths
import torch
from .config import Config
from .trainer import Trainer
from .utils import detect_task_spooler


def cli():
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--anno", default=None)
    parser.add_argument("--img", default=None)
    parser.add_argument("--seg", default=None)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--workers", default=12, type=int)
    parser.add_argument("--model-state", default=None)
    parser.add_argument("--no-bpbar", default=False, action="store_true")
    args = parser.parse_args()

    if args.anno is None:
        anno_path = project_paths.psg_annotation_dir
        print("Anno Path:", anno_path)
    else:
        anno_path = args.anno
    if args.img is None:
        img_dir = project_paths.psg_img_dir
        print("Img Dir:", img_dir)
    else:
        img_dir = args.img
    if args.seg is None:
        seg_dir = project_paths.psg_seg_dir
        print("Seg Dir:", seg_dir)
    else:
        seg_dir = args.seg

    config = Config.from_file(args.config)

    if args.model_state is None:
        model_state_dict = None
    else:
        model_state_dict = torch.load(args.model_state, map_location="cpu")["model"]

    if args.no_bpbar:
        hide_batch_progress = True
    elif detect_task_spooler():
        print("Detected task-spooler. Batch progress will be hidden")
        hide_batch_progress = True
    else:
        hide_batch_progress = False

    trainer = Trainer(
        anno_path=anno_path,
        img_dir=img_dir,
        seg_dir=seg_dir,
        out_dir=args.output,
        config=config,
        num_workers=args.workers,
        start_state_dict=model_state_dict,
        hide_batch_progress=hide_batch_progress,
    )
    if trainer.out_dir:
        with open(trainer.out_dir / "args.json", "w") as f:
            json.dump(args.__dict__, f)
    trainer.run(epochs=args.epochs)
    # to easily check which experiments ran to the end
    if trainer.out_dir:
        with open(trainer.out_dir / "done.txt", "w") as f:
            f.write("done")

    print(
        "Best value for ",
        trainer.critical_metric,
        ": ",
        trainer.best_metric_value,
        sep="",
    )


if __name__ == "__main__":
    cli()
