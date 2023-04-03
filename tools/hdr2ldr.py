import cv2
import os


root_path = "/home/SecondDisk/Code/ours/InverseHouse/exps/Mat-hdr11/2022_10_22_08_26_15/plots/novel_view"


# all_items = sorted(os.listdir(root_path), key= lambda x: int(os.path.splitext(x)[0].split('_')[-2]))

# for item in all_items:
#     full_path = os.path.join(root_path, item)

#     img = cv2.imread(full_path, -1)
#     img = img**(1/2.2)
#     cv2.imwrite(full_path.replace('.hdr','.png'), img*255)

root_path = root_path.replace('novel_view','editing')
all_items = sorted(os.listdir(root_path), key= lambda x: int(os.path.splitext(x)[0].split('_')[-2]))

for item in all_items:
    full_path = os.path.join(root_path, item)

    img = cv2.imread(full_path, -1)
    img = img**(1/2.2)
    cv2.imwrite(full_path.replace('.hdr','.png'), img*255)


# cmd = ""