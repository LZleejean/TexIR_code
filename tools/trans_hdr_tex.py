import cv2
import numpy as np
import os
import glob


def getFiles(root):
    with open(os.path.join(root),'r') as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    return lines


def repackHDRTexture(projectPath):

    scanIdList = getFiles(os.path.join(projectPath, 'info/aligned.txt'))

    fileList = glob.glob(os.path.join(projectPath,'hdr_texture','0.png'))
    for fileName in fileList:

        img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)    
        # height, width, bands = img.shape
        hdrMap = np.zeros(img.shape, dtype=np.float32)

        hdrIdList = np.unique(img[:, :, 2])
        for hdrId in hdrIdList:

            # if hdrId == 0:
            #     continue

            rows, cols = np.where(img[:, :, 2] == hdrId)
            scanId = scanIdList[hdrId]
            hdrFile = os.path.join(projectPath,'hdr', '%s/ccm.hdr' % scanId)

            hdr = cv2.imread(hdrFile, cv2.IMREAD_UNCHANGED)
            height, width, bands = hdr.shape
            #hdr = hdrFill(hdr)
            
            # depth_img_fix_path = os.path.join(projectPath, 'derived', '%s/depth_image_fix.png' % scanId)
            # depth_img_fix = cv2.imread(depth_img_fix_path, cv2.IMREAD_UNCHANGED)
            # depth_img_fix = cv2.resize(depth_img_fix, (width, height))
            # depth_img_fix = np.asarray(depth_img_fix, np.float32)
            # depth_img_fix = np.repeat(depth_img_fix[...,np.newaxis], 3, axis=-1)/5000.0
            # inv_attenuation = (1. + 0.14 * depth_img_fix + 0.07*depth_img_fix*depth_img_fix)
            
            
            
            hdrCol = (img[rows, cols, 1] / 50000 * width).astype(np.int)
            hdrCol = np.clip(hdrCol, 0, width - 1)
            hdrRow = (img[rows, cols, 0] / 50000 * height).astype(np.int)
            hdrRow = np.clip(hdrRow, 0, height - 1)
            hdrMap[rows, cols, :] = hdr[hdrRow.astype(np.int), hdrCol.astype(np.int), :3] #* inv_attenuation[hdrRow.astype(np.int), hdrCol.astype(np.int)]
            
            h_seams, w_seams = np.where((img[:,:,0]+img[:,:,1]+img[:,:,2]) == 0)
            hdrMap[h_seams, w_seams] = np.array([0,0,0],dtype=np.float32)[np.newaxis,:]

        cv2.imwrite(fileName.replace('.png', '_hdr_ccm_upper.hdr'), hdrMap)

    return True


def repackSegTexture(projectPath):

    scanIdList = getFiles(os.path.join(projectPath, 'info/aligned.txt'))

    fileList = glob.glob(os.path.join(projectPath,'hdr_texture','0.png'))
    for fileName in fileList:

        img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
        # height, width, bands = img.shape
        hdrMap = np.zeros(list(img.shape), dtype=np.uint8)

        hdrIdList = np.unique(img[:, :, 2])
        for hdrId in hdrIdList:

            # if hdrId == 0:
            #     continue

            rows, cols = np.where(img[:, :, 2] == hdrId)
            scanId = scanIdList[hdrId]
            hdrFile = os.path.join(projectPath,'derived', '%s/panoImage_gray.png' % scanId)

            hdr = cv2.imread(hdrFile, cv2.IMREAD_UNCHANGED)
            hdr = np.repeat(hdr[:,:,np.newaxis], 3, axis=-1)
            height, width, c = hdr.shape
            #hdr = hdrFill(hdr)
            
            # depth_img_fix_path = os.path.join(projectPath, 'derived', '%s/depth_image_fix.png' % scanId)
            # depth_img_fix = cv2.imread(depth_img_fix_path, cv2.IMREAD_UNCHANGED)
            # depth_img_fix = cv2.resize(depth_img_fix, (width, height))
            # depth_img_fix = np.asarray(depth_img_fix, np.float32)
            # depth_img_fix = np.repeat(depth_img_fix[...,np.newaxis], 3, axis=-1)/5000.0
            # inv_attenuation = (1. + 0.14 * depth_img_fix + 0.07*depth_img_fix*depth_img_fix)
            
            
            
            hdrCol = (img[rows, cols, 1] / 50000 * width).astype(np.int)
            hdrCol = np.clip(hdrCol, 0, width - 1)
            hdrRow = (img[rows, cols, 0] / 50000 * height).astype(np.int)
            hdrRow = np.clip(hdrRow, 0, height - 1)
            hdrMap[rows, cols, :] = hdr[hdrRow.astype(np.int), hdrCol.astype(np.int), :] #* inv_attenuation[hdrRow.astype(np.int), hdrCol.astype(np.int)]
            
            h_seams, w_seams = np.where((img[:,:,0]+img[:,:,1]+img[:,:,2]) == 0)
            hdrMap[h_seams, w_seams] = np.array([0,0,0],dtype=np.uint8)[np.newaxis,:]

        cv2.imwrite(fileName.replace('.png', '_seg_gray.png'), hdrMap)

    return True

def repackAlbedoTexture(projectPath, methods='phyir'):

    scanIdList = getFiles(os.path.join(projectPath, 'info/aligned.txt'))

    fileList = glob.glob(os.path.join(projectPath,'hdr_texture','0.png'))
    for fileName in fileList:

        img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
        # height, width, bands = img.shape
        hdrMap = np.zeros(list(img.shape), dtype=np.uint8)

        hdrIdList = np.unique(img[:, :, 2])
        for hdrId in hdrIdList:

            # if hdrId == 0:
            #     continue

            rows, cols = np.where(img[:, :, 2] == hdrId)
            scanId = scanIdList[hdrId]
            hdrFile = os.path.join(projectPath,'methods/{}'.format(methods), '%s/albedo.png' % scanId)

            hdr = cv2.imread(hdrFile, cv2.IMREAD_UNCHANGED)
            if len(hdr.shape) == 2:
                hdr = np.repeat(hdr[:,:,np.newaxis], 3, axis=-1)
            height, width, c = hdr.shape
            #hdr = hdrFill(hdr)
            
            # depth_img_fix_path = os.path.join(projectPath, 'derived', '%s/depth_image_fix.png' % scanId)
            # depth_img_fix = cv2.imread(depth_img_fix_path, cv2.IMREAD_UNCHANGED)
            # depth_img_fix = cv2.resize(depth_img_fix, (width, height))
            # depth_img_fix = np.asarray(depth_img_fix, np.float32)
            # depth_img_fix = np.repeat(depth_img_fix[...,np.newaxis], 3, axis=-1)/5000.0
            # inv_attenuation = (1. + 0.14 * depth_img_fix + 0.07*depth_img_fix*depth_img_fix)
            
            
            
            hdrCol = (img[rows, cols, 1] / 50000 * width).astype(np.int)
            hdrCol = np.clip(hdrCol, 0, width - 1)
            hdrRow = (img[rows, cols, 0] / 50000 * height).astype(np.int)
            hdrRow = np.clip(hdrRow, 0, height - 1)
            hdrMap[rows, cols, :] = (hdr[hdrRow.astype(np.int), hdrCol.astype(np.int), :] / 255.0) **(1/2.2) *255.0 #* inv_attenuation[hdrRow.astype(np.int), hdrCol.astype(np.int)]
            
            h_seams, w_seams = np.where((img[:,:,0]+img[:,:,1]+img[:,:,2]) == 0)
            hdrMap[h_seams, w_seams] = np.array([0,0,0],dtype=np.uint8)[np.newaxis,:]

        hdrMap = cv2.resize(hdrMap, (2048,2048))
        kernel = np.ones((4,4), np.uint8)
        hdrMap = cv2.dilate(hdrMap, kernel)

        cv2.imwrite(fileName.replace('.png', '_{}_albedo.png'.format(methods)), hdrMap)

    return True

def repackRoughnessTexture(projectPath, methods='phyir'):

    scanIdList = getFiles(os.path.join(projectPath, 'info/aligned.txt'))

    fileList = glob.glob(os.path.join(projectPath,'hdr_texture','0.png'))
    for fileName in fileList:

        img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
        # height, width, bands = img.shape
        hdrMap = np.zeros(list(img.shape), dtype=np.uint8)

        hdrIdList = np.unique(img[:, :, 2])
        for hdrId in hdrIdList:

            # if hdrId == 0:
            #     continue

            rows, cols = np.where(img[:, :, 2] == hdrId)
            scanId = scanIdList[hdrId]
            hdrFile = os.path.join(projectPath,'methods/{}'.format(methods), '%s/roughness.png' % scanId)

            hdr = cv2.imread(hdrFile, cv2.IMREAD_UNCHANGED)
            if len(hdr.shape) == 2:
                hdr = np.repeat(hdr[:,:,np.newaxis], 3, axis=-1)
            height, width, c = hdr.shape
            #hdr = hdrFill(hdr)
            
            # depth_img_fix_path = os.path.join(projectPath, 'derived', '%s/depth_image_fix.png' % scanId)
            # depth_img_fix = cv2.imread(depth_img_fix_path, cv2.IMREAD_UNCHANGED)
            # depth_img_fix = cv2.resize(depth_img_fix, (width, height))
            # depth_img_fix = np.asarray(depth_img_fix, np.float32)
            # depth_img_fix = np.repeat(depth_img_fix[...,np.newaxis], 3, axis=-1)/5000.0
            # inv_attenuation = (1. + 0.14 * depth_img_fix + 0.07*depth_img_fix*depth_img_fix)
            
            
            
            hdrCol = (img[rows, cols, 1] / 50000 * width).astype(np.int)
            hdrCol = np.clip(hdrCol, 0, width - 1)
            hdrRow = (img[rows, cols, 0] / 50000 * height).astype(np.int)
            hdrRow = np.clip(hdrRow, 0, height - 1)
            hdrMap[rows, cols, :] = (hdr[hdrRow.astype(np.int), hdrCol.astype(np.int), :] / 255.0) **(1/2.2) *255.0 #* inv_attenuation[hdrRow.astype(np.int), hdrCol.astype(np.int)]
            
            h_seams, w_seams = np.where((img[:,:,0]+img[:,:,1]+img[:,:,2]) == 0)
            hdrMap[h_seams, w_seams] = np.array([0,0,0],dtype=np.uint8)[np.newaxis,:]

        hdrMap = cv2.resize(hdrMap, (2048,2048))
        kernel = np.ones((4,4), np.uint8)
        hdrMap = cv2.dilate(hdrMap, kernel)

        cv2.imwrite(fileName.replace('.png', '_{}_roughness.png'.format(methods)), hdrMap)

    return True

if __name__=="__main__":
    # repackHDRTexture("/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/customHouse/vrproc")
    repackSegTexture("/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/customHouse/vrproc")
    # repackAlbedoTexture("/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/cyclops-8RDdZ75Q1l0Moa3L_1-47070f9567484e67c7d920785ba0a21c/vrproc",'phyir')
    # repackRoughnessTexture("/home/SecondDisk/test_data/galois_model/inverse/hdrhouse/cyclops-8RDdZ75Q1l0Moa3L_1-47070f9567484e67c7d920785ba0a21c/vrproc",'phyir')



    ## trans predictions into derived id, for texture
    lists = [
        'cyclops-nPOdM9m9ByLM53j4_1',
        'cyclops-5KrO2owbWbO2oake_1-6d934463938c506bdaa0c1d40f35d5be',
        'cyclops-41DjM6BrN3vMqlzG_1',
        'cyclops-Rk3OMA7A6b4Zx5dl_1-8327f6e5b39fc9c0ea09949871dafb01',
        'cyclops-wDJqZg3yexXZA36e_1',
        'cyclops-XlDNZQNxobx2AgxK_1-9234975438f259283279f24d29141a59',
        'cyclops-XKoGMD5peyYV536D_1',
        'cyclops-XlDNZQNxgjE2AgxK_1-2c4eb6326ec7eb0785f818a71b02bd6e',
        'cyclops-8RDdZ75Q1l0Moa3L_1-47070f9567484e67c7d920785ba0a21c',
        None,
        'cyclops-7AGzV3bqKNRMXe6Q_2-4829c2937400a719f3bc85a22bd10085'
    ]

    method_name = 'nvdiffrec'

    root_path = "/home/SecondDisk/test_data/galois_model/inverse/hdrhouse"

    for i in range(len(lists)):
        if not i ==10:
            continue
        # if lists[i]==None:
        #     continue
        repackAlbedoTexture(os.path.join(root_path, lists[i], 'vrproc'), method_name)
        repackRoughnessTexture(os.path.join(root_path, lists[i], 'vrproc'), method_name)

        target_path = "/home/SecondDisk/CVPR2023/comparison/{}/hdrhouse{}/mesh/albedo".format(method_name,i+1)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        cmd = 'mv -f {} {}'.format(os.path.join(root_path, lists[i], 'vrproc', 'hdr_texture', "0_{}_albedo.png".format(method_name)), target_path)
        os.system(cmd)
        # cmd = "cp -f {} {}".format(os.path.join(root_path, lists[i], 'vrproc', 'hdr_texture', "out1.obj"), target_path)
        # os.system(cmd)
        # cmd = "cp -f {} {}".format(os.path.join(root_path, lists[i], 'vrproc', 'hdr_texture', "output.mtl"), target_path)
        # os.system(cmd)

        target_path = "/home/SecondDisk/CVPR2023/comparison/{}/hdrhouse{}/mesh/roughness".format(method_name, i+1)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        cmd = 'mv -f {} {}'.format(os.path.join(root_path, lists[i], 'vrproc', 'hdr_texture', "0_{}_roughness.png".format(method_name)), target_path)
        os.system(cmd)
        # cmd = "cp -f {} {}".format(os.path.join(root_path, lists[i], 'vrproc', 'hdr_texture', "out1.obj"), target_path)
        # os.system(cmd)
        # cmd = "cp -f {} {}".format(os.path.join(root_path, lists[i], 'vrproc', 'hdr_texture', "output.mtl"), target_path)
        # os.system(cmd)


    # # generate seg texture
    # lists = [
    #     'cyclops-nPOdM9m9ByLM53j4_1',
    #     'cyclops-5KrO2owbWbO2oake_1-6d934463938c506bdaa0c1d40f35d5be',
    #     'cyclops-41DjM6BrN3vMqlzG_1',
    #     'cyclops-Rk3OMA7A6b4Zx5dl_1-8327f6e5b39fc9c0ea09949871dafb01',
    #     'cyclops-wDJqZg3yexXZA36e_1',
    #     'cyclops-XlDNZQNxobx2AgxK_1-9234975438f259283279f24d29141a59',
    #     'cyclops-XKoGMD5peyYV536D_1',
    #     'cyclops-XlDNZQNxgjE2AgxK_1-2c4eb6326ec7eb0785f818a71b02bd6e',
    #     'cyclops-8RDdZ75Q1l0Moa3L_1-47070f9567484e67c7d920785ba0a21c',
    #     None,
    #     'cyclops-7AGzV3bqKNRMXe6Q_2-4829c2937400a719f3bc85a22bd10085'
    # ]

    # root_path = "/home/SecondDisk/test_data/galois_model/inverse/hdrhouse"

    # for i in range(len(lists)):
    #     if i ==8:
    #         continue
    #     if lists[i]==None:
    #         continue
    #     repackSegTexture(os.path.join(root_path, lists[i], 'vrproc'))