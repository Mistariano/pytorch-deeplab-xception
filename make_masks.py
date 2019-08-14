#
# demo.py
#
import argparse
import os
import numpy as np

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-dir', type=str, required=True, help='image to test')
    parser.add_argument('--in-flist', type=str, required=True, help='image flist')
    parser.add_argument('--out-dir', type=str, required=True, help='mask image to save')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    return args


def prepare():
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(args.in_flist, 'r') as f:
        in_flist = f.readlines()
    in_flist = [f.split(' ')[0] for f in in_flist]
    return in_flist, args


def make_mask(in_flist, model, args):
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    pbar = tqdm(in_flist, total=len(in_flist))
    for fname in pbar:
        outpath = os.path.join(args.out_dir, fname)
        if os.path.exists(outpath):
            pbar.set_description(fname + ' exists')
            continue
        fpath = os.path.join(args.in_dir, fname)
        pbar.set_description('')
        image = Image.open(fpath).convert('RGB')
        target = image.convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        grid_image = make_grid(
            decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
            3, normalize=False, range=(0, 255))
        grid_image = grid_image[0:1, ...] + grid_image[1:2, ...] + grid_image[2:3, ...]
        grid_image[grid_image > 0] = 255

        # area = ((grid_image / 255.).sum()) / grid_image.shape[0] / grid_image.shape[1]

        _outdir = os.path.split(outpath)[:-1]
        _outdir = '/'.join(_outdir)
        if not os.path.exists(_outdir):
            os.makedirs(_outdir)

        assert grid_image.shape[0] == image.size[0]
        assert grid_image.shape[1] == image.size[1]
        save_image(grid_image, outpath)

        # yield area


def get_model(args):
    model = DeepLab(num_classes=21,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    if args.cuda:
        model.cuda()


def task_loop(task_queue: Queue, model, args, log_queue: Queue = None):
    while True:
        task = task_queue.get()
        if not task:
            task_queue.put(None)
            break

        make_mask(task, model, args)
        if log_queue is not None:
            log_queue.put(len(task))


def process_bar_loop(queue: Queue, total):
    pbar = tqdm(range(total), total=total)
    while True:
        cnt = queue.get()
        if cnt is None:
            break
        pbar.update(cnt)


def scheduler(flist, args):
    num_workers = args.num_workers
    models = [get_model(args) for _ in range(num_workers)]

    total_task_cnt = len(flist)
    size_task = 8
    num_task = total_task_cnt // size_task

    def _get_tasks():
        for i in range(num_task):
            task = flist[i * size_task:i * size_task + size_task]
            yield task
        if total_task_cnt % size_task != 0:
            task = flist[total_task_cnt // size_task * size_task:]
            yield task

    queue = Queue()
    log_queue = Queue()
    log_process = Process(target=process_bar_loop, args=[queue, total_task_cnt])
    log_process.start()
    pool = Pool()
    for model in models:
        pool.apply_async(task_loop, args=[queue, model, args], kwds={'log_queue': log_queue})
    for task in _get_tasks():
        # add task
        queue.put(task)
    queue.put(None)
    pool.join()
    log_queue.put(None)
    log_process.join()


def main():
    tasks, args = prepare()
    scheduler(tasks, args)


if __name__ == "__main__":
    main()
