import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # required by GeForce RTX 3090
import math
import time
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDPGAN
from collections import OrderedDict
import utils
import dataset
from network_PeQuENet_QPAdaptation import PeQuENet
import torch.nn as nn
import torch.nn.functional as F
from vgg import vgg19
from vgg import AntiAliasInterpolation2d

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class discriminator(nn.Module):

    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        out = {}
        feature_maps = []
        x1 = F.leaky_relu(self.conv1(input), 0.2)
        feature_maps.append(x1)
        x2 = F.leaky_relu(self.conv2_bn(self.conv2(x1)), 0.2)
        feature_maps.append(x2)
        x3 = F.leaky_relu(self.conv3_bn(self.conv3(x2)), 0.2)
        feature_maps.append(x3)
        x4 = F.leaky_relu(self.conv4_bn(self.conv4(x3)), 0.2)
        feature_maps.append(x4)
        x = self.conv5(x4)
        out['feature_maps'] = feature_maps
        out['prediction'] = x
        return out

class ImagePyramide(torch.nn.Module):

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

def receive_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='3frames_mfqev2_1G.yml',
        help='Path to option YAML file.'
        )
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )

    # suggest to use only 1 GPU to ensure you can successfully run the codes
    opts_dict['train']['num_gpu'] = torch.cuda.device_count()

    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = False
    else:
        opts_dict['train']['is_dist'] = False
    
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )

    return opts_dict

def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    # we sometimes calculate PNSR (dB) during the training, but we don't optimize PSNR.
    # So the values of PSNR will not influence the network training.
    # We optimize perceptual quality with the help of GAN. loss = G_loss + 10 * feature_loss + 10 * vgg_loss
    unit = opts_dict['train']['criterion']['unit'] # criterion: PSNR, unit: dB
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])
    
    # ==========
    # init distributed training
    # ==========

    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank,
            backend='nccl'
            )

    # TO-DO: load resume states if exists
    pass

    # ==========
    # create logger
    # ==========

    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'w')

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========

    pass
    
    # ==========
    # fix random seed
    # ==========

    seed = opts_dict['train']['random_seed']
    # >I don't know why should rs + rank
    utils.set_random_seed(seed + rank)

    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create train and val data prefetchers
    # ==========
    
    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    val_ds_type = opts_dict['dataset']['val']['type']
    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    assert val_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    val_ds_cls = getattr(dataset, val_ds_type)
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'], 
        radius=radius
        )
    val_ds = val_ds_cls(
        opts_dict=opts_dict['dataset']['val'], 
        radius=radius
        )

    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds, 
        num_replicas=opts_dict['train']['num_gpu'], 
        rank=rank, 
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
        )
    val_sampler = None  # no need to sample val data

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds, 
        opts_dict=opts_dict, 
        sampler=train_sampler, 
        phase='train',
        seed=opts_dict['train']['random_seed']
        )
    val_loader = utils.create_dataloader(
        dataset=val_ds, 
        opts_dict=opts_dict, 
        sampler=val_sampler, 
        phase='val'
        )
    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * \
        opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * \
        opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    val_num = len(val_ds)
    
    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    # ==========
    # create model
    # ==========

    model = PeQuENet()

    model = model.to(rank)
    if opts_dict['train']['is_dist']:
        model = DDP(model, device_ids=[rank])

    lr = 1e-4
    D = discriminator(32)
    D.weight_init(mean=0.0, std=0.02)
    D.cuda()
    D_optimizer = torch.optim.Adam(D.parameters(), lr, betas=(0.9, 0.999))


    assert opts_dict['train']['loss'].pop('type') == 'CharbonnierLoss', \
        "Not implemented."
    # loss_func = utils.CharbonnierLoss(**opts_dict['train']['loss'])

    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    optimizer = optim.Adam(
        model.parameters(),
        **opts_dict['train']['optim']
        )

    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == \
            'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        scheduler = utils.CosineAnnealingRestartLR(
            optimizer, 
            **opts_dict['train']['scheduler']
            )
        opts_dict['train']['scheduler']['is_on'] = True

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()


    start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"val sequence: [{val_num}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # evaluate original performance, e.g., PSNR before enhancement
    # ==========

    vid_num = val_ds.get_vid_num()
    if opts_dict['train']['pre-val'] and rank == 0:
        msg = f"\n{'<' * 10} Pre-evaluation {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        per_aver_dict = {}
        for i in range(vid_num):
            per_aver_dict[i] = utils.Counter()
        pbar = tqdm(
                total=val_num, 
                ncols=opts_dict['train']['pbar_len']
                )

        # fetch the first batch
        val_prefetcher.reset()
        val_data = val_prefetcher.next()

        while val_data is not None:
            # get data
            gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = val_data['lq'].to(rank)  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            b, _, _, _, _  = lq_data.shape
            
            # eval
            batch_perf = np.mean(
                [criterion(lq_data[i,radius,...], gt_data[i]) for i in range(b)]
                )  # bs must be 1!
            
            # log
            per_aver_dict[index_vid].accum(volume=batch_perf)

            # display
            pbar.set_description(
                "{:s}: [{:.3f}] {:s}".format(name_vid, batch_perf, unit)
                )
            pbar.update()

            # fetch next batch
            val_data = val_prefetcher.next()

        pbar.close()

        # log
        ave_performance = np.mean([
            per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
            ])
        msg = "> ori performance: [{:.3f}] {:s}".format(ave_performance, unit)
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_timer = utils.Timer()  # total tra + val time of each epoch

    # ==========
    # start training + validation (test)
    # ==========

    model.train()
    num_iter_accum = start_iter


    # add VGG model
    vgg = Vgg19()
    if torch.cuda.is_available():
        vgg = vgg.cuda()
    scales = [1, 0.5, 0.25, 0.125]
    pyramid = ImagePyramide(scales, num_channels= 1)
    if torch.cuda.is_available():
        pyramid = pyramid.cuda()
    perceptual_weights= [0.03125, 0.0625, 0.125, 0.25, 1]


    for current_epoch in range(start_epoch, num_epoch + 1):
        # shuffle distributed subsamplers before each epoch
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()

        # train this epoch
        while train_data is not None:

            # over sign
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # get data
            gt_data = train_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = train_data['lq'].to(rank)  # (B T [RGB] H W)
            qp_data = train_data['qp'].to(rank)  # (B T [RGB] H W)
            b, _, c, _, _  = lq_data.shape
            input_data = torch.cat(
                [lq_data[:,:,i,...] for i in range(c)],
                dim=1
                )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            enhanced_data = model(radius, input_data, qp_data)

            D_fake_decision = D(enhanced_data.detach())
            D_fake_loss = torch.mean(D_fake_decision['prediction'].squeeze() ** 2) / 2
            D_real_decision = D(gt_data)
            D_real_loss = torch.mean((D_real_decision['prediction'].squeeze() - 1) ** 2) / 2
            D_loss = D_real_loss + D_fake_loss

            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            D_result = D(enhanced_data)
            D_real = D(gt_data)

            G_loss = torch.mean((D_result['prediction'].squeeze() - 1) ** 2)

            # add feature matching loss
            feature_loss = 0
            for i, (a, b) in enumerate(zip(D_real['feature_maps'], D_result['feature_maps'])):
                value = torch.abs(a - b).mean()
                feature_loss += value

            # add vgg loss
            pyramide_real = pyramid(gt_data)
            pyramide_generated = pyramid(enhanced_data)
            vgg_loss = 0
            for scale in scales:
                x_vgg = vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(perceptual_weights):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    vgg_loss += perceptual_weights[i] * value

            optimizer.zero_grad()  # zero grad
            loss = G_loss + 10 * feature_loss + 10 * vgg_loss

            loss.backward()  # cal grad
            optimizer.step()  # update parameters

            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if (num_iter_accum % interval_print == 0) and (rank == 0):
                # display & log

                lr = optimizer.param_groups[0]['lr']
                loss_item = loss.item()
                G_loss_item = G_loss.item()
                D_loss_item = D_loss.item()
                feature_loss_item = feature_loss.item()
                vgg_loss_item = vgg_loss.item()
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    "lr: [{:.3f}]x1e-4, loss: [{:.4f}], G_loss: [{:.4f}], D_loss: [{:.4f}], feature_loss: [{:.4f}], vgg_loss: [{:.4f}]".format(
                        lr*1e4, loss_item, G_loss_item, D_loss_item, feature_loss_item, vgg_loss_item
                        )
                    )
                print(msg)
                log_fp.write(msg + '\n')


            if ((num_iter_accum % interval_val == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    ".pt"
                    )
                state = {
                    'num_iter_accum': num_iter_accum,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)

                # save Discriminator model
                checkpoint_save_path_DN = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    "_discriminator"
                    ".pt"
                )
                state_DN = {
                    'num_iter_accum': num_iter_accum,
                    'state_dict': D.state_dict(),
                    'optimizer': D_optimizer.state_dict(),
                }

                torch.save(state_DN, checkpoint_save_path_DN)


                # validation
                with torch.no_grad():
                    per_aver_dict = {}
                    for index_vid in range(vid_num):
                        per_aver_dict[index_vid] = utils.Counter()
                    pbar = tqdm(
                            total=val_num,
                            ncols=opts_dict['train']['pbar_len']
                            )

                    # train -> eval
                    model.eval()

                    # fetch the first batch
                    val_prefetcher.reset()
                    val_data = val_prefetcher.next()

                    while val_data is not None:
                        # get data
                        gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
                        lq_data = val_data['lq'].to(rank)  # (B T [RGB] H W)
                        index_vid = val_data['index_vid'].item()
                        name_vid = val_data['name_vid'][0]  # bs must be 1!
                        b, _, c, _, _  = lq_data.shape
                        input_data = torch.cat(
                            [lq_data[:,:,i,...] for i in range(c)],
                            dim=1
                            )
                        qp_num = torch.tensor([2]).unsqueeze(0).to(0)
                        enhanced_data = model(radius, input_data, qp_num)  # (B [RGB] H W)

                        # eval
                        batch_perf = np.mean(
                            [criterion(enhanced_data[i], gt_data[i]) for i in range(b)]
                            ) # bs must be 1!

                        # display
                        pbar.set_description(
                            "{:s}: [{:.3f}] {:s}"
                            .format(name_vid, batch_perf, unit)
                            )
                        pbar.update()

                        # log
                        per_aver_dict[index_vid].accum(volume=batch_perf)

                        # fetch next batch
                        val_data = val_prefetcher.next()

                    # end of val
                    pbar.close()

                    # eval -> train
                    model.train()

                # log
                ave_per = np.mean([
                    per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
                    ])
                msg = (
                    "> model saved at {:s}\n"
                    "> ave val per: [{:.3f}] {:s}"
                    ).format(
                        checkpoint_save_path, ave_per, unit
                        )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()

        # end of this epoch (training dataloader exhausted)

    # end of all epochs

    # ==========
    # final log & close logger
    # ==========

    if rank == 0:
        total_time = total_timer.get_interval() / 3600
        msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
        print(msg)
        log_fp.write(msg + '\n')
        
        msg = (
            f"\n{'<' * 10} Goodbye {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        
        log_fp.close()


if __name__ == '__main__':
    main()
