import torch
import numpy as np
from collections import OrderedDict
from network_PeQuENet_QPAdaptation import PeQuENet
import utils

# !! please change to your path !!
ckp_path = 'F:/PeQuENet/exp/MFQEv2_R3_enlarge300x/ckp_model.pt' # model path
rec_yuv_save_path = 'G:/QP37/' # enhanced video path (output path)
cmp_yuv_path = 'F:/HM_encode_test_3231/HEVC_QP37_3231' # compressed video path (input path)

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
                  'PartyScene_832x480_500.yuv','BasketballDrill_832x480_500.yuv',
                  'BQSquare_416x240_600.yuv', 'RaceHorses_832x480_300.yuv',
                  'FourPeople_1280x720_600.yuv', 'KristenAndSara_1280x720_600.yuv',
                  'Johnny_1280x720_600.yuv', 'BasketballDrive_1920x1080_500.yuv',
                  'Kimono_1920x1080_240.yuv','BQTerrace_1920x1080_600.yuv',
                  'Cactus_1920x1080_500.yuv','ParkScene_1920x1080_240.yuv',                  
                  'Traffic_2560x1600_150.yuv','PeopleOnStreet_2560x1600_150.yuv']

    for video in video_list:
        lq_yuv_path = cmp_yuv_path + '/' + video

        h = int(video.split('_', 2)[1].split('x')[1])
        w = int(video.split('_', 2)[1].split('x')[0])
        nfs = int(video.split('_', 2)[2].split('.')[0])

        msg = f'loading low-quality yuv...'
        print(msg)

        lq_y, lq_u, lq_v = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
        )

        lq_y = lq_y.astype(np.float32) / 255.
        lq_u = lq_u.reshape(nfs, 1, -1)
        lq_v = lq_v.reshape(nfs, 1, -1)
        lq_uv = np.concatenate((lq_u, lq_v), axis=2)
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
            # How to split the input_data depends on the sequence resolution and your CUDA memory. 
            # For example, when implemented on NVIDIA 2080ti, we do not need to split sequences in Class C and D.
            # But we split sequences in Class A as
            # input_data1 = input_data[:, :, :, :256]
            # input_data2 = input_data[:, :, :, 256:256*2]
            # input_data3 = input_data[:, :, :, 256*2:256*3]
            # input_data4 = input_data[:, :, :, 256*3:256*4]
            # input_data5 = input_data[:, :, :, 256*4:256*5]
            # input_data6 = input_data[:, :, :, 256*5:256*6]
            # input_data7 = input_data[:, :, :, 256*6:256*7]
            # input_data8 = input_data[:, :, :, 256*7:256*8]
            # input_data9 = input_data[:, :, :, 256*8:256*9]
            # input_data10 = input_data[:, :, :,256*9:256*10]
            # For sequences in Class B:
            # input_data1 = input_data[:, :, :, :320]
            # input_data2 = input_data[:, :, :, 320:320*2]
            # input_data3 = input_data[:, :, :, 320*2:320*3]
            # input_data4 = input_data[:, :, :, 320*3:320*4]
            # input_data5 = input_data[:, :, :, 320*4:320*5]
            # input_data6 = input_data[:, :, :, 320*5:320*6]
            # For sequences in Class E:
            # input_data1 = input_data[:, :, :, :640]
            # input_data2 = input_data[:, :, :, 640:]
            

            # enhance
            with torch.no_grad():
                # torch.tensor([0]): QP 22 torch.tensor([1]): QP 27
                # torch.tensor([2]): QP 32 torch.tensor([3]): QP 37
                qp_num = torch.tensor([3]).unsqueeze(0).to(0)
                enhanced_frm = model(opts_dict['radius'], input_data, qp_num)
                # For sequences in Class A, we enhance each part indepently and then merge them. 
                # enhanced_frm_1 = model(opts_dict['radius'], input_data1, qp_num)
                # enhanced_frm_2 = model(opts_dict['radius'], input_data2, qp_num)
                # enhanced_frm_3 = model(opts_dict['radius'], input_data3, qp_num)
                # enhanced_frm_4 = model(opts_dict['radius'], input_data4, qp_num)
                # enhanced_frm_5 = model(opts_dict['radius'], input_data5, qp_num)
                # enhanced_frm_6 = model(opts_dict['radius'], input_data6, qp_num)
                # enhanced_frm_7 = model(opts_dict['radius'], input_data7, qp_num)
                # enhanced_frm_8 = model(opts_dict['radius'], input_data8, qp_num)
                # enhanced_frm_9 = model(opts_dict['radius'], input_data9, qp_num)
                # enhanced_frm_10 = model(opts_dict['radius'], input_data10, qp_num)
                # enhanced_frm = torch.cat((enhanced_frm_1,enhanced_frm_2,enhanced_frm_3,
                #                     enhanced_frm_4,enhanced_frm_5,enhanced_frm_6,
                #                     enhanced_frm_7,enhanced_frm_8,enhanced_frm_9,
                #                     enhanced_frm_10),dim=3)
                # For sequences in Class B:
                # enhanced_frm_1 = model(opts_dict['radius'], input_data1, qp_num)
                # enhanced_frm_2 = model(opts_dict['radius'], input_data2, qp_num)
                # enhanced_frm_3 = model(opts_dict['radius'], input_data3, qp_num)
                # enhanced_frm_4 = model(opts_dict['radius'], input_data4, qp_num)
                # enhanced_frm_5 = model(opts_dict['radius'], input_data5, qp_num)
                # enhanced_frm_6 = model(opts_dict['radius'], input_data6, qp_num)
                # enhanced_frm = torch.cat((enhanced_frm_1,enhanced_frm_2,enhanced_frm_3,
                #                     enhanced_frm_4,enhanced_frm_5,enhanced_frm_6),dim=3)
                # For sequences in Class E:
                # enhanced_frm_1 = model(opts_dict['radius'], input_data1, qp_num)
                # enhanced_frm_2 = model(opts_dict['radius'], input_data2, qp_num)
                # enhanced_frm = torch.cat((enhanced_frm_1,enhanced_frm_2),dim=3)

            data = enhanced_frm.detach().cpu().numpy()
            data = np.clip(data, 0, 1)
            enhanced_frm_uint8 = np.squeeze((data * 255).astype('uint8'))
            enhanced_frame[idx, :, :] = np.squeeze(enhanced_frm_uint8)

        enhanced_frame = enhanced_frame.reshape(nfs, 1, -1)
        enhanced_yuv = np.concatenate((enhanced_frame, lq_uv), axis=2)
        enhanced_yuv = enhanced_yuv.flatten()

        fp = open(rec_yuv_save_path + video, 'wb+')
        fp.write(enhanced_yuv)
        fp.close()
        print('one yuv done.')

    print('> done.')


if __name__ == '__main__':
    main()
