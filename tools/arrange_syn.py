import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image


# total 49 views, 49-11=38, 38-14=24.
# 11 views
skip_list = [
    18,#
    25,#
    32,
    39,
    40,
    41,
    42,
    46,
    47,
    48,
    49
]

# 14 views
novel_view = [
    1,
    2,
    6,
    9,
    11,
    13,
    16,
    17,
    #19,#
    20,
    22,
    #25,
    27,
    30,
    34,
    38
]

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
palette = [0, 0, 0, 128, 0, 0, 192, 0, 0, 128, 128, 0, 0, 128, 0, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                64, 0, 0, 0, 0, 128, 64, 128, 0, 64, 128, 192, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128,
                128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 64, 128, 64,
                192, 128, 64, 192, 0, 192, 192, 128, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64,
                192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0,
                128,
                192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 0, 192, 128, 128, 192, 128, 64, 0, 192, 64, 64, 0,
                192, 64, 0, 192, 128, 192, 0, 64, 64, 0, 64, 104]


root_path = "/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/customHouse/mr_fur_optix"
target_root_path = "/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/customHouse/vrproc"

all_items = glob.glob(os.path.join(root_path, '00[0-9]*'))
all_items.sort(key= lambda x: int(os.path.split(x)[1]))

target_derived_root_path = os.path.join(target_root_path, 'derived')
target_info_path = os.path.join(target_root_path, 'info')
target_hdr_root_path = os.path.join(target_root_path, 'hdr')
if not os.path.exists(target_derived_root_path):
    os.makedirs(target_derived_root_path)
if not os.path.exists(target_info_path):
    os.makedirs(target_info_path)
if not os.path.exists(target_hdr_root_path):
    os.makedirs(target_hdr_root_path)

# generate derived
for i in range(len(all_items)):
    if i+1 in skip_list:
        continue
    target_derived_path = os.path.join(target_derived_root_path, '{}'.format(i+1))
    if not os.path.exists(target_derived_path):
        os.makedirs(target_derived_path)
    target_hdr_path = os.path.join(target_hdr_root_path, '{}'.format(i+1))
    if not os.path.exists(target_hdr_path):
        os.makedirs(target_hdr_path)
    
    # generate color
    img = cv2.imread(os.path.join(all_items[i], 'scene_ldr.png'))
    h, w, c = img.shape
    # img = cv2.resize(img, (6720,3360))
    cv2.imwrite(os.path.join(target_derived_path, 'panoImage.jpg'), img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(os.path.join(target_derived_path, 'panoImage_orig.png'), img)
    os.rename(os.path.join(target_derived_path, 'panoImage_orig.png'), os.path.join(target_derived_path, 'panoImage_orig.jpg'))
    
    # # read depth
    # depth_array = np.fromfile(os.path.join(root_path, 'render', 'scene_alldepth_{}.dat'.format(i+1)), np.float32)
    # depth_img = depth_array[2:].reshape(h, w)
    # depth_img = cv2.resize(depth_img, (1600,800))
    # depth_img = depth_img*5000
    # depth_img = np.asarray(depth_img, np.uint16)
    # cv2.imwrite(os.path.join(target_derived_path, 'depth_image.png'), depth_img)

    # generate segmentation from basecolor
    # basecolor = cv2.imread(os.path.join(root_path, 'render', 'scene_allbaseColor_{}.png'.format(i+1)), -1)[:,:,::-1]    # bgr2rgb
    # h, w, c = basecolor.shape
    # seg_gray = np.zeros(basecolor.shape, dtype=np.uint8)
    # basecolor = torch.from_numpy(basecolor.copy())
    # seg_gray = torch.from_numpy(seg_gray.copy())
    # # floor
    # seg_gray = torch.where(basecolor==torch.tensor([255,0,0], dtype=torch.uint8), torch.tensor([46,46,46], dtype=torch.uint8), seg_gray)
    # # wall
    # seg_gray = torch.where(basecolor==torch.tensor([216,216,216], dtype=torch.uint8), torch.tensor([45,45,45], dtype=torch.uint8), seg_gray)
    # # door
    # seg_gray = torch.where(basecolor==torch.tensor([186,186,186], dtype=torch.uint8), torch.tensor([1,1,1], dtype=torch.uint8), seg_gray)
    # # ceiling
    # seg_gray = torch.where(basecolor==torch.tensor([243,243,243], dtype=torch.uint8), torch.tensor([43,43,43], dtype=torch.uint8), seg_gray)
    # # lamp
    # seg_gray = torch.where(basecolor==torch.tensor([230,230,230], dtype=torch.uint8), torch.tensor([27,27,27], dtype=torch.uint8), seg_gray)
    # # gray sphere -> 26
    # seg_gray = torch.where(basecolor==torch.tensor([155,160,159], dtype=torch.uint8), torch.tensor([26,26,26], dtype=torch.uint8), seg_gray)
    # # yellow sphere -> 25
    # seg_gray = torch.where(basecolor==torch.tensor([247,225,74], dtype=torch.uint8), torch.tensor([25,25,25], dtype=torch.uint8), seg_gray)
    # # blue sphere -> 24
    # seg_gray = torch.where(basecolor==torch.tensor([54,138,208], dtype=torch.uint8), torch.tensor([24,24,24], dtype=torch.uint8), seg_gray)
    # # green sphere -> 23
    # seg_gray = torch.where(basecolor==torch.tensor([82,199,82], dtype=torch.uint8), torch.tensor([23,23,23], dtype=torch.uint8), seg_gray)
    # seg_color = colorize_mask(seg_gray.numpy()[:,:,0], palette)
    # cv2.imwrite(os.path.join(target_derived_path, 'panoImage_gray.png'), seg_gray.numpy()[:,:,0])
    # seg_color.save(os.path.join(target_derived_path, 'panoImage_seg.png'))

    # # generate basecolor
    # basecolor = cv2.imread(os.path.join(root_path, 'render', 'scene_allbaseColor_{}.png'.format(i+1)), -1)
    # cv2.imwrite(os.path.join(target_derived_path, 'albedo.png'), basecolor)
    # generate roughness
    roughness = cv2.imread(os.path.join(root_path, 'render', 'scene_allroughness_{}.png'.format(i+1)), -1)
    cv2.imwrite(os.path.join(target_derived_path, 'roughness.png'), roughness)
    # # generate normal
    # normal = cv2.imread(os.path.join(root_path, 'render', 'scene_allnormal_{}.png'.format(i+1)), -1)
    # cv2.imwrite(os.path.join(target_derived_path, 'normal.png'), normal)

    # generate hdr
    img = cv2.imread(os.path.join(all_items[i], 'scene_hdr.hdr'),-1)
    img = img*(2**(-5)) # note that we scale the intensity
    cv2.imwrite(os.path.join(target_hdr_path, 'ccm.hdr'), img)



# # generate info
# aligned_path = os.path.join(target_info_path, 'aligned.txt')
# with open(aligned_path, 'w') as f:
#     for i in range(len(all_items)):
#         if i+1 in skip_list:
#             continue
#         if i+1 in novel_view:
#             continue
#         f.write('{}'.format(i+1))
#         if not i == len(all_items)-1:
#             f.write('\n')
# with open(os.path.join(target_info_path, 'novel.txt'), 'w') as f:
#     for i in range(len(all_items)):
#         if i+1 in novel_view:
#             f.write('{}'.format(i+1))
#             if not i == len(all_items)-1:
#                 f.write('\n')

# with open(os.path.join(root_path, 'cameraFile_optix.txt'), 'r') as f:
#     lines = f.readlines()
# # fit cyclops's final extrinsics.
# lines = [i.replace(' \n', '\n') for i in lines]
# extrinsics = np.loadtxt(lines[1:], delimiter=' ')   # shape: (N*4, 4)

# with open(os.path.join(target_info_path, 'final_extrinsics.txt'), 'w') as f:
#     f.write("{}\n".format(len(all_items)-len(skip_list)-len(novel_view)))
#     for i in range(len(all_items)):
#         if i+1 in skip_list:
#             continue
#         if i+1 in novel_view:
#             continue
#         # f.write(lines[i*4+1:(i+1)*4+1])
#         camera_position = lines[i*3+1]
#         camera_position = camera_position.strip().split(' ')
#         # ensure consistency for vrproc (-y as positive)
#         up = [0.0, -1.0, 0.0]
#         front = [0.0, 0.0, 1.0]
#         right = [-1.0, 0.0, 0.0]
#         f.write("{} {} {} {}\n".format(right[0], up[0], front[0], -float(camera_position[0])))
#         f.write("{} {} {} {}\n".format(right[1], up[1], front[1], float(camera_position[1])))
#         f.write("{} {} {} {}\n".format(right[2], up[2], front[2], -float(camera_position[2])))
#         f.write("0 0 0 1\n")


# with open(os.path.join(target_info_path, 'novel_extrinsics.txt'), 'w') as f:
#     f.write("{}\n".format(len(novel_view)))
#     for i in range(len(all_items)):
#         if i+1 in novel_view:
#             # f.write(lines[i*4+1:(i+1)*4+1])
#             camera_position = lines[i*3+1]
#             camera_position = camera_position.strip().split(' ')
#             # ensure consistency for vrproc (-y as positive)
#             up = [0.0, -1.0, 0.0]
#             front = [0.0, 0.0, 1.0]
#             right = [-1.0, 0.0, 0.0]
#             f.write("{} {} {} {}\n".format(right[0], up[0], front[0], -float(camera_position[0])))
#             f.write("{} {} {} {}\n".format(right[1], up[1], front[1], float(camera_position[1])))
#             f.write("{} {} {} {}\n".format(right[2], up[2], front[2], -float(camera_position[2])))
#             f.write("0 0 0 1\n")