import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import tqdm

import utils
from network_PeQuENet_QPAdaptation import PeQuENet

VIDEO_LIST = [
    # Class A
    # {'name': 'Traffic_2560x1600_150.yuv', 'crop': (None, 256)},
    # {'name': 'PeopleOnStreet_2560x1600_150.yuv', 'crop': (None, 256)},
    # Class B
    {'name': 'ParkScene_1920x1080_240.yuv', 'crop': (None, 320)},
    {'name': 'Kimono_1920x1080_240.yuv', 'crop': (None, 320)},
    {'name': 'BQTerrace_1920x1080_600.yuv', 'crop': (None, 320)},
    {'name': 'Cactus_1920x1080_500.yuv', 'crop': (None, 320)},
    {'name': 'BasketballDrive_1920x1080_500.yuv', 'crop': (None, 320)},
    # Class C
    {'name': 'BasketballDrill_832x480_500.yuv', 'crop': (None, None)},
    {'name': 'BQMall_832x480_600.yuv', 'crop': (None, None)},
    {'name': 'PartyScene_832x480_500.yuv', 'crop': (None, None)},
    {'name': 'RaceHorses_832x480_300.yuv', 'crop': (None, None)},
    # Class D
    {'name': 'BasketballPass_416x240_500.yuv', 'crop': (None, None)},
    {'name': 'BlowingBubbles_416x240_500.yuv', 'crop': (None, None)},
    {'name': 'RaceHorses_416x240_300.yuv', 'crop': (None, None)},
    {'name': 'BQSquare_416x240_600.yuv', 'crop': (None, None)},
    # Class E
    {'name': 'FourPeople_1280x720_600.yuv', 'crop': (None, 640)},
    {'name': 'KristenAndSara_1280x720_600.yuv', 'crop': (None, 640)},
    {'name': 'Johnny_1280x720_600.yuv', 'crop': (None, 640)},
]


def main(args):
    print("> Using device: %d" % args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    opts_dict = {
        'radius': 1
    }
    model = PeQuENet()
    msg = f'> Loading model: {args.ckp_path}'
    print(msg)

    checkpoint = torch.load(args.ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model = model.cuda()
    model.eval()

    utils.mkdir(args.rec_yuv_save_path)

    for video_meta in VIDEO_LIST:

        video = video_meta["name"]
        crop = video_meta["crop"]

        lq_yuv_path = os.path.join(args.cmp_yuv_path, video)
        raw_yuv_path = os.path.join(args.raw_yuv_base_path, video)

        try:
            if args.width is None: args.width = int(video.split('_', 2)[1].split('x')[0])
            if args.height is None: args.height = int(video.split('_', 2)[1].split('x')[1])
            if args.num_frames is None: args.num_frames = int(video.split('_', 2)[2].split('.')[0])
        except:
            raise (Exception("Data from filename not detected"))

        if args.crop_size == (0, 0):  # do not crop
            crop = None
        if args.crop_size == None:  # read from video list
            args.crop_size = crop
        if args.crop_size[0] == None:  # crop by columns
            args.crop_size = (args.width, crop[1])
        if args.crop_size[1] == None:  # crop by rows
            args.crop_size = (crop[0], args.height)

        msg = "> Loading video: %s" % video
        print(msg)

        in_range = (2 ** args.input_bit_depth - 1)
        out_range = (2 ** args.output_bit_depth - 1)
        out_dt = np.dtype('uint8') if args.output_bit_depth == 8 else np.dtype('<u2')

        lq_y, lq_u, lq_v = utils.import_yuv(
            fname=lq_yuv_path, height=args.height, width=args.width, num_frames=args.num_frames,
            frame_skip=0, bitdepth=args.input_bit_depth
        )
        raw_y, raw_u, raw_v = utils.import_yuv(
            fname=raw_yuv_path, height=args.height, width=args.width, num_frames=args.num_frames,
            frame_skip=0, bitdepth=args.input_bit_depth
        )

        lq_y = lq_y.astype(np.float32) / in_range

        raw_u = raw_u.reshape(args.num_frames, 1, -1)
        raw_v = raw_v.reshape(args.num_frames, 1, -1)
        raw_uv = np.concatenate((raw_u, raw_v), axis=2)
        raw_uv = raw_uv.astype(np.float32) / in_range
        raw_uv = (raw_uv * out_range).astype(out_dt)

        enhanced_frame = np.zeros((args.num_frames, args.height, args.width), dtype=out_dt)

        for idx in tqdm.tqdm(range(args.num_frames)):

            idx_list = list(range(idx - opts_dict['radius'], idx + opts_dict['radius'] + 1))
            idx_list = np.clip(idx_list, 0, args.num_frames - 1)

            input_data = []
            for idx_ in idx_list:
                input_data.append(lq_y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0)
            input_data = input_data.cuda()

            # enhance
            qp_num = torch.tensor([args.qp]).unsqueeze(0).to(0)
            with torch.no_grad():
                if crop is None:
                    enhanced_frm = model(opts_dict['radius'], input_data, qp_num)
                else:
                    patches = utils.get_patches(input_data, args.crop_size)

                    enhanced = []
                    for row in patches:
                        enhanced_row = []
                        for patch in row:
                            enhanced_row.append(model(opts_dict['radius'], patch, qp_num))
                        enhanced.append(enhanced_row)

                    enhanced_frm = utils.combine_patches(enhanced)

            data = enhanced_frm.detach().cpu().numpy()
            data = np.clip(data, 0, 1)
            enhanced_frm_uint8 = np.squeeze((data * out_range).astype(out_dt))
            enhanced_frame[idx, :, :] = np.squeeze(enhanced_frm_uint8)

        enhanced_frame = enhanced_frame.reshape(args.num_frames, 1, -1)
        enhanced_yuv = np.concatenate((enhanced_frame, raw_uv), axis=2)
        enhanced_yuv = enhanced_yuv.flatten()

        fp = open("%s/%s" % (args.rec_yuv_save_path, video), 'wb+')
        fp.write(enhanced_yuv)
        fp.close()

    print('> done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test PeQuENet")
    parser.add_argument("--ckp_path", type=str, default="./model/ckp_model.pt", help="model path")
    parser.add_argument("--rec_yuv_save_path", type=str, default="./output/test",
                        help="path to output enhanced sequences")
    parser.add_argument("--cmp_yuv_path", type=str, default="./sequences/compressed",
                        help="path to compressed sequences")
    parser.add_argument("--raw_yuv_base_path", type=str, default="./sequences/raw",
                        help="path to raw sequences")
    parser.add_argument("--device", type=int, default=0, help="number of gpu device")
    parser.add_argument("--input_bit_depth", type=int, default=8, help="compressed video bit depth")
    parser.add_argument("--output_bit_depth", type=int, default=8, help="enhanced video bit depth")
    parser.add_argument("--width", type=int, default=None, help="video width, if None get from file name")
    parser.add_argument("--height", type=int, default=None, help="video height, if None get from file name")
    parser.add_argument("--num_frames", type=int, default=None, help="video frames, if None get from file name")
    parser.add_argument("--crop_size", type=int, default=None,
                        help="crop size in the tuple form (width, height), "
                             "if (0, 0) do not crop, if None read from video list, "
                             "if (None, height) crop by columns, "
                             "if (width, None) crop by rows")
    parser.add_argument("--qp", type=int, default=3,
                        help="0: 22, 1: 27, 2: 32, 3: 37")

    main(parser.parse_args())
