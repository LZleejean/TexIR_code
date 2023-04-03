'''
@File    :   Pano2Cube.py
@Time    :   2023/02/27 12:07:51
@Author  :   Zhen Li 
@Contact :   yodlee@mail.nwpu.edu.cn
@Institution: Realsee
@License :   GNU General Public License v2.0
@Desc    :   None
'''



import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as tranforms



class Pano2Cube:
    """
    Differentiable panorama2cubemap op. It supports gpu and batch processing.
    """
    def __init__(self,batch_size=1,pano_width = 256,pano_height=128,cube_lenth=128,cube_channel =3,is_cuda=False) -> None:
        
        self.pano_width = pano_width
        self.pano_height = pano_height
        self.batch_size = batch_size
        self.cube_lenth = cube_lenth
        self.cube_channel = cube_channel
        self.is_cuda = is_cuda
        # face order: 0-left,1-front,2-right,3-back,4-top,5-bottom
        self.horizon_angle = np.array([-90.0,0.0,90.0,180.0,])/180.0*np.pi
        self.vertical_angle = np.array([-90.0,90.0])/180.0*np.pi
        self.rorate_list = []
        for  h in self.horizon_angle:
            temp = h*np.array([0,1,0],np.float32)      # rotate as y axis
            rotate_vector,_ = cv2.Rodrigues(temp)
            self.rorate_list.append(rotate_vector)
        
        for v in self.vertical_angle:
            temp = v*np.array([1,0,0],np.float32)      # rotate as x axis
            rotate_vector,_ = cv2.Rodrigues(temp)
            self.rorate_list.append(rotate_vector)

        self.rorate_list = [Variable(torch.FloatTensor(x)) for x in self.rorate_list]

        scl = 1.0
        sample_x, sample_y = np.meshgrid(
            np.linspace(-scl, scl, cube_lenth),
            np.linspace(scl, -scl, cube_lenth)
        )

        r = np.sqrt(sample_y * sample_y + sample_x * sample_x + 1)
        sample_x /= r
        sample_y /= r
        sample_z = np.sqrt(1 - sample_y * sample_y - sample_x * sample_x)
        
        # xyz = torch.cat([Variable(torch.FloatTensor(sample_x)),Variable(torch.FloatTensor(sample_y)),Variable(torch.FloatTensor(sample_z))],dim=2)
        # xyz = xyz.view(cube_lenth*cube_lenth,3).permute(1,0)    # size: (3,cube_lenth*cube_lenth)
        # xyz = torch.FloatTensor([sample_x,sample_y,sample_z])
        xyz = torch.from_numpy(np.array([sample_x,sample_y,sample_z], dtype=np.float32))
        xyz = xyz.view(3,cube_lenth*cube_lenth)
        self.uv = []
        for i,R_matrix in enumerate(self.rorate_list):
            temp_xyz = torch.matmul(R_matrix,xyz).permute(1,0)  # size: (cube_lenth*cube_lenth,3)
            # convert to polar
            azimuth = torch.atan2(temp_xyz[:,0], temp_xyz[:,2])  # [-Pi, Pi]
            elevation = torch.asin(temp_xyz[:,1])  # [-Pi/2, Pi/2]
            azimuth = azimuth.view(1,cube_lenth,cube_lenth,1)
            elevation = elevation.view(1,cube_lenth,cube_lenth,1)
            # convert to uv, because grid_sample's grid value range:[-1,1]. -1 correspoding left-top,1 corresponding right-down
            # self.u = (azimuth+np.pi)*1/np.pi/2
            # self.v = (np.pi-elevation)*1/np.pi
            u = azimuth/np.pi
            v = -elevation/(np.pi/2)

            self.uv.append(torch.cat([u.repeat(batch_size,1,1,1),v.repeat(batch_size,1,1,1)],dim=3))
        
    def Tocube(self,input,mode='bilinear'):
        """
        pano 2 cube.

        Args:
            input (torch.float32): shape: [b, c, h, w]
            mode (str, optional): interpolation mode (nearest, bilinear). Defaults to 'bilinear'.

        Returns:
            torch.float32: shape: [b, c*6, h, w]
        """
        assert mode in ['bilinear', 'nearest']
        out = []
        for i in range(6):
            uv = self.uv[i].cuda() if self.is_cuda else self.uv[i]
            result = F.grid_sample(input,uv,mode=mode,padding_mode="border",align_corners=False) # there is one problem, for back face(theta = +-pi), not warp interperation.scipy.ndimage.map_coordinates has this op but its not differential
            out.append(result)
        out = torch.cat(out,dim=1)  #size:(batch_size,6*3,out,out)
        return out



    # def ToCubeTensor(self, batch, mode='bilinear'):
    #     assert mode in ['bilinear', 'nearest']
    #     batch_size = batch.size()[0]
    #     out_batch = self.Tocube(batch, mode=mode)
    #     out_batch = torch.cat(out_batch,dim=1)  # batch_size,channel*6,h,w
        
    #     return out_batch
    
    def cube2full(self,batch):
        out = torch.zeros([self.batch_size,self.cube_channel,self.cube_lenth*3,self.cube_lenth*4])
        # id:0
        out[:,:,self.cube_lenth:self.cube_lenth*2,0:self.cube_lenth] = batch[:,0:3,:,:]
        out[:,:,self.cube_lenth:self.cube_lenth*2,self.cube_lenth:self.cube_lenth*2] = batch[:,3:6,:,:]
        out[:,:,self.cube_lenth:self.cube_lenth*2,self.cube_lenth*2:self.cube_lenth*3] = batch[:,6:9,:,:]
        out[:,:,self.cube_lenth:self.cube_lenth*2,self.cube_lenth*3:self.cube_lenth*4] = batch[:,9:12,:,:]
        out[:,:,0:self.cube_lenth,self.cube_lenth:self.cube_lenth*2] = batch[:,12:15,:,:]
        out[:,:,self.cube_lenth*2:self.cube_lenth*3,self.cube_lenth:self.cube_lenth*2] = batch[:,15:18,:,:]

        return out
    
    def saveCube(self,path,result,model="full"):    # define batch_size=1
        fname,file_type = os.path.splitext(path)
        
        if model =="full":
            result = result.permute(0,2,3,1)
            for i in range(result.size()[0]):
                one = np.asarray(result[i],dtype=np.float32)
                cv2.imwrite(fname+str(i)+file_type,one*255)
        elif model=="6":
            for i in range(6):
                one = result[:,i*self.cube_channel:(i+1)*self.cube_channel,:,:].permute(0,2,3,1)
                one = np.asarray(one[0].cpu(),dtype=np.float32)
                cv2.imwrite(fname+"_"+str(i)+file_type,one*255)


if __name__=="__main__":


    #-----------------------------
    
    path = "/home/SecondDisk/Code/colleague/reproject/derived/1646101296/panoImage.jpg"
    img = cv2.imread(path,cv2.IMREAD_UNCHANGED)[:,:,0:3]
    h,w,c= img.shape
    # img = rotate_pano(img,0,0,0)
    img = np.asarray(img,dtype=np.float32)
    fname,file_type = os.path.splitext(path)
    if(file_type == ".hdr" or file_type == ".exr"):
            # img = img*pow(2,3.4)
            img = img**(1/2.2)
    else:
        img = img/255.0
    img = tranforms.ToTensor()(img)
    
    img = torch.reshape(img,(1,c,h,w))
    img = img.repeat(1,1,1,1).cuda()
    trans = Pano2Cube(batch_size=img.size()[0],pano_height=h,pano_width=w,is_cuda=True,cube_lenth=1024)
    
    # out = trans.Tocube(img)
    # # out = trans.ToCubeTensor(img)
    # for i in range(6):
    #     one = out[i].permute(0,2,3,1)
    #     one = np.asarray(one[0],dtype=np.float32)

    #     cv2.imwrite("test_cube_{}.jpg".format(i),one*255)

    # trans.saveCube("test_cube_full.png",trans.cube2full(trans.Tocube(img)))
    trans.saveCube("/home/SecondDisk/Code/colleague/reproject/derived/1646101296/cube.jpg",trans.Tocube(img),model="6")

