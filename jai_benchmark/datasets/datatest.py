from .dataset_base import DatasetBase

class ZzfDb(DatasetBase):
    def __init__(self, download=False, dest_dir=None, num_frames=None, name=None, **kwargs):
        super().__init__(num_frames=num_frames, name=name, **kwargs)




a = ZzfDb(num_frames=100, name='zzf')