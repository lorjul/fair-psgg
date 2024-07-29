import socket
from pathlib import Path


class PathContainer:
    psg_img_dir: Path
    psg_seg_dir: Path
    psg_annotation_dir: Path

    def __init__(self, **kwargs):
        self.paths = kwargs

    def __getattr__(self, item):
        p = self.paths.get(item, None)
        if p is None:
            raise KeyError(f"Project path not set for: {item}")
        return Path(p)


project_paths = PathContainer()

current_host_name = socket.gethostname()
# use current_host_name to decide how to override PathContainer here
if current_host_name == "your_host":
    project_paths = PathContainer(
        psg_img_dir="/path/to/psg/coco",
        psg_seg_dir="/path/to/psg/coco",
        psg_annotation_dir="/path/to/psg/psg/psg.json",
    )
