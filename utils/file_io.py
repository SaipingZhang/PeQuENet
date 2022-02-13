import numpy as np

# ==========
# YUV420P
# ==========


def import_yuv(fname, width, height, num_frames=0, frame_skip=0, bitdepth=8):
    yuvfile = open(fname, mode='rb')

    n_elements_luma = width * height
    n_elements_chroma = int(n_elements_luma / 4)
    n_elements_frame = (n_elements_luma + 2 * n_elements_chroma)
    nBytes = 1 if bitdepth == 8 else 2
    dtRead = np.dtype('uint8') if bitdepth == 8 else np.dtype('<u2')
    dt = np.dtype('uint8') if bitdepth == 8 else np.dtype('<u2')
    skip_offset = 0
    if frame_skip > 0:
        print('Skipping frames at the beginning (if necessary)')
        skip_offset = int(n_elements_frame * nBytes)

        # Try to read the whole thing at once:
    num_items = -1
    if num_frames > 0:
        num_items = int(n_elements_frame * num_frames)

    data = np.fromfile(yuvfile, dtype=dtRead, count=num_items, offset=skip_offset)
    n_frames = int(data.shape[0] / n_elements_frame)

    luma = np.zeros(shape=(n_frames, height, width,), dtype=dt)
    chroma_u = np.zeros(shape=(n_frames, int(height / 2), int(width / 2)), dtype=dt)
    chroma_v = np.zeros(shape=(n_frames, int(height / 2), int(width / 2)), dtype=dt)

    offset = 0
    # arrange data in frames
    for f in range(0, n_frames):
        luma_array = data[offset:offset + n_elements_luma]
        luma[f, :, :] = np.reshape(np.transpose(luma_array), (height, width))
        offset = offset + n_elements_luma

        chroma_array = data[offset:offset + n_elements_chroma]
        chroma_u[f, :, :] = np.reshape(np.transpose(chroma_array), (int(height / 2), int(width / 2)))
        offset = offset + n_elements_chroma

        chroma_array = data[offset:offset + n_elements_chroma]
        chroma_v[f, :, :] = np.reshape(np.transpose(chroma_array), (int(height / 2), int(width / 2)))
        offset = offset + n_elements_chroma
    return luma, chroma_u, chroma_v


def write_ycbcr(y, cb, cr, vid_path):
    with open(vid_path, 'wb') as fp:
        for ite_frm in range(len(y)):
            fp.write(y[ite_frm].reshape(((y[0].shape[0])*(y[0].shape[1]), )))
            fp.write(cb[ite_frm].reshape(((cb[0].shape[0])*(cb[0].shape[1]), )))
            fp.write(cr[ite_frm].reshape(((cr[0].shape[0])*(cr[0].shape[1]), )))


# ==========
# FileClient
# ==========

class _HardDiskBackend():
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf


class _LmdbBackend():
    """Lmdb storage backend.

    Args:
        db_path (str): Lmdb database path.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.

    Attributes:
        db_paths (str): Lmdb database path.
    """
    def __init__(self,
                 db_paths,
                 client_keys='default',
                 readonly=True,
                 lock=False,
                 readahead=False,
                 **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        assert len(client_keys) == len(self.db_paths), (
            'client_keys and db_paths should have the same length, '
            f'but received {len(client_keys)} and {len(self.db_paths)}.')

        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(
                path,
                readonly=readonly,
                lock=lock,
                readahead=readahead,
                **kwargs)

    def get(self, filepath, client_key):
        """Get values according to the filepath from one lmdb named client_key.
        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
            client_key (str): Used for distinguishing differnet lmdb envs.
        """
        filepath = str(filepath)
        assert client_key in self._client, (f'client_key {client_key} is not '
                                            'in lmdb clients.')
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf


class FileClient(object):
    """A file client to access LMDB files or general files on disk.
    
    Return a binary file."""

    _backends = {
        'disk': _HardDiskBackend,
        'lmdb': _LmdbBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend == 'disk':
            self.client = _HardDiskBackend()
        elif backend == 'lmdb':
            self.client = _LmdbBackend(**kwargs)
        else:
            raise ValueError(f'Backend {backend} not supported.')
        self.backend = backend

    def get(self, filepath, client_key='default'):
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

# ==========
# Dict
# ==========

def dict2str(input_dict, indent=0):
    """Dict to string for printing options."""
    msg = ''
    indent_msg = ' ' * indent
    for k, v in input_dict.items():
        if isinstance(v, dict):  # still a dict
            msg += indent_msg + k + ':[\n'
            msg += dict2str(v, indent+2)
            msg += indent_msg + '  ]\n'
        else:  # the last level
            msg += indent_msg + k + ': ' + str(v) + '\n'
    return msg

# ==========
# Dataloader prefetcher
# ==========
import torch
import threading
import queue as Queue
from torch.utils.data import DataLoader


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Ref:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Ref:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher():
    """CUDA prefetcher.

    Ref:
    https://github.com/NVIDIA/apex/issues/304#

    It may consums more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
