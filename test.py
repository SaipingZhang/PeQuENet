import torch
import numpy as np
from collections import OrderedDict
from network_PeQuENet_QPAdaptation import PeQuENet
import utils

# !! please change to your path !!
ckp_path = 'F:/PeQuENet/exp/MFQEv2_R3_enlarge300x/ckp_model.pt' # model path
rec_yuv_save_path = 'G:/QP37/' # enhanced video path (output path)
cmp_yuv_path = 'F:/HM_encode_test_3231/HEVC_QP37_3231' # compressed video path (input path)
raw_yuv_base_path = 'F:/raw' # raw video (video before compression) path

def main():

    opts_dict = {
        'radius': 1
        }
    model = PeQuENet()
    msg = f'loading model {ckp_path}...'
    print(msg)

    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training, recommend!
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    video_list = ['RaceHorses_416x240_300.yuv','BlowingBubbles_416x240_500.yuv',
                  'BasketballPass_416x240_500.yuv','BQMall_832x480_600.yuv',
                  'BQSquare_416x240_600.yuv','FourPeople_1280x720_600.yuv',
                  'Johnny_1280x720_600.yuv','KristenAndSara_1280x720_600.yuv',
                  'PartyScene_832x480_500.yuv','BasketballDrill_832x480_500.yuv',
                  'Kimono_1920x1080_240.yuv','BQTerrace_1920x1080_600.yuv',
                  'Cactus_1920x1080_500.yuv','ParkScene_1920x1080_240.yuv',
                  'RaceHorses_832x480_300.yuv','BasketballDrive_1920x1080_500.yuv',
                  'Traffic_2560x1600_150.yuv','PeopleOnStreet_2560x1600_150.yuv']

    for video in video_list:
        lq_yuv_path = cmp_yuv_path + '/' + video
        raw_yuv_path = raw_yuv_base_path + '/' + video

        h = int(video.split('_', 2)[1].split('x')[1])
        w = int(video.split('_', 2)[1].split('x')[0])
        nfs = int(video.split('_', 2)[2].split('.')[0])

        msg = f'loading raw and low-quality yuv...'
        print(msg)
        raw_y, raw_u, raw_v = utils.import_yuv(
            seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
        )
        lq_y = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )

        lq_y = lq_y.astype(np.float32) / 255.
        raw_u = raw_u.reshape(nfs, 1, -1)
        raw_v = raw_v.reshape(nfs, 1, -1)
        raw_uv = np.concatenate((raw_u, raw_v), axis=2)
        print(raw_uv.shape)
        msg = '> yuv loaded.'
        print(msg)

        enhanced_frame = np.zeros((nfs, h, w), dtype='uint8')

        for idx in range(nfs):

            idx_list = list(range(idx - opts_dict['radius'], idx + opts_dict['radius'] + 1))
            idx_list = np.clip(idx_list, 0, nfs - 1)

            input_data = []
            for idx_ in idx_list:
                input_data.append(lq_y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0)
            input_data = input_data.cuda()

            # If you do not have enough memory, you may need to split the input_data
            # into few parts and enhance each part and merge them finally.

            # enhance
            with torch.no_grad():
                # torch.tensor([0]): QP 22 torch.tensor([1]): QP 27
                # torch.tensor([2]): QP 32 torch.tensor([3]): QP 33
                qp_num = torch.tensor([3]).unsqueeze(0).to(0)
                enhanced_frm = model(opts_dict['radius'], input_data, qp_num)

            data = enhanced_frm.detach().cpu().numpy()
            data = np.clip(data, 0, 1)
            enhanced_frm_uint8 = np.squeeze((data * 255).astype('uint8'))
            enhanced_frame[idx, :, :] = np.squeeze(enhanced_frm_uint8)

        enhanced_frame = enhanced_frame.reshape(nfs, 1, -1)
        enhanced_yuv = np.concatenate((enhanced_frame, raw_uv), axis=2)
        enhanced_yuv = enhanced_yuv.flatten()

        fp = open(rec_yuv_save_path + video, 'wb+')
        fp.write(enhanced_yuv)
        fp.close()
        print('one yuv done.')

    print('> done.')


if __name__ == '__main__':
    main()
