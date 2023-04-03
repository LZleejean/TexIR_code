'''
@File    :   Cube2Pano.py
@Time    :   2023/02/27 12:07:32
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



class Cube2Pano:
    """
    Differentiable cubemap2panorama op. It supports gpu and batch processing.
    """
    def __init__(self,batch_size=1,pano_width = 256,pano_height=128,cube_lenth=128,cube_channel =3,is_cuda=False, cube_padding_size=0) -> None:
        
        self.pano_width = pano_width
        self.pano_height = pano_height
        self.batch_size = batch_size
        self.cube_lenth = cube_lenth
        self.cube_channel = cube_channel
        self.is_cuda = is_cuda
        self.cube_padding_size = cube_padding_size

        step = 2*np.pi/pano_width
        sample_theta,sample_phi = np.meshgrid(
            np.linspace(-np.pi,np.pi,pano_width,dtype=np.float32),
            np.linspace(0.5*np.pi,-0.5*np.pi,pano_height,dtype=np.float32)
        )
        sample_theta = torch.from_numpy(sample_theta)
        sample_phi = torch.from_numpy(sample_phi)

        self.x = torch.cos(sample_phi)*torch.sin(sample_theta)
        self.y = torch.sin(sample_phi)
        self.z = torch.cos(sample_phi)*torch.cos(sample_theta)

        self.grid , self.mask = self.getCubeGrid()
        self.grid = self.grid.repeat(batch_size,1,1,1)  #size:(6*batch_size,h,w,2)
        self.mask = self.mask.repeat(batch_size,1,1,1)  #size:(6*batch_size,h,w,1)

        if self.is_cuda:
            self.grid = self.grid.cuda()
            self.mask = self.mask.cuda()

    
    def getCubeGrid(self):
        self.sphere_coor = torch.stack([self.x,self.y,self.z],dim=2)
        
        # in order to determine pano mask, just divide abs(x or y or z) and then judge positive or negnative
        # left
        grid_left_x = torch.abs(self.x).view(self.pano_height,self.pano_width,1).expand(-1,-1,3)
        grid_left_tmp = self.sphere_coor/grid_left_x
        grid_left_u = grid_left_tmp[:,:,2]
        grid_left_v = -grid_left_tmp[:,:,1] # convert to uv, because grid_sample's grid value range:[-1,1]. -1 correspoding left-top,1 corresponding right-down
        grid_left = torch.stack([grid_left_u,grid_left_v],dim=2)
        mask_left = ((grid_left_u)>=-1) * ((grid_left_u)<=1) * ((grid_left_v)>=-1) * ((grid_left_v)<=1) * ((grid_left_tmp[:,:,0]<0))
        mask_left = mask_left.unsqueeze(-1).float()
        
        # front
        grid_front_z = torch.abs(self.z).view(self.pano_height,self.pano_width,1).expand(-1,-1,3)
        grid_front_tmp = self.sphere_coor/grid_front_z
        grid_front_u = grid_front_tmp[:,:,0]
        grid_front_v = -grid_front_tmp[:,:,1] # convert to uv, because grid_sample's grid value range:[-1,1]. -1 correspoding left-top,1 corresponding right-down
        grid_front = torch.stack([grid_front_u,grid_front_v],dim=2)
        mask_front = ((grid_front_u)>=-1) * ((grid_front_u)<=1) * ((grid_front_v)>=-1) * ((grid_front_v)<=1) * ((grid_front_tmp[:,:,2]>0))
        mask_front = mask_front.unsqueeze(-1).float()

        # right
        grid_right_x = torch.abs(self.x).view(self.pano_height,self.pano_width,1).expand(-1,-1,3)
        grid_right_tmp = self.sphere_coor/grid_right_x
        grid_right_u = -grid_right_tmp[:,:,2]
        grid_right_v = -grid_right_tmp[:,:,1] # convert to uv, because grid_sample's grid value range:[-1,1]. -1 correspoding left-top,1 corresponding right-down
        grid_right = torch.stack([grid_right_u,grid_right_v],dim=2)
        mask_right = ((grid_right_u)>=-1) * ((grid_right_u)<=1) * ((grid_right_v)>=-1) * ((grid_right_v)<=1) * ((grid_right_tmp[:,:,0]>0.0))
        mask_right = mask_right.unsqueeze(-1).float()
        

        # back
        grid_back_z = torch.abs(self.z).view(self.pano_height,self.pano_width,1).expand(-1,-1,3)
        grid_back_tmp = self.sphere_coor/grid_back_z
        grid_back_u = -grid_back_tmp[:,:,0]
        grid_back_v = -grid_back_tmp[:,:,1] # convert to uv, because grid_sample's grid value range:[-1,1]. -1 correspoding left-top,1 corresponding right-down
        grid_back = torch.stack([grid_back_u,grid_back_v],dim=2)
        mask_back = ((grid_back_u)>=-1) * ((grid_back_u)<=1) * ((grid_back_v)>=-1) * ((grid_back_v)<=1) * ((grid_back_tmp[:,:,2]<0))
        mask_back = mask_back.unsqueeze(-1).float()

        # top
        grid_top_y = torch.abs(self.y).view(self.pano_height,self.pano_width,1).expand(-1,-1,3)
        grid_top_tmp = self.sphere_coor/grid_top_y
        grid_top_u = grid_top_tmp[:,:,0]
        grid_top_v = grid_top_tmp[:,:,2] # convert to uv, because grid_sample's grid value range:[-1,1]. -1 correspoding left-top,1 corresponding right-down
        grid_top = torch.stack([grid_top_u,grid_top_v],dim=2)
        mask_top = ((grid_top_u)>=-1) * ((grid_top_u)<=1) * ((grid_top_v)>=-1) * ((grid_top_v)<=1) * ((grid_top_tmp[:,:,1]>0.0))
        mask_top = mask_top.unsqueeze(-1).float()

        # bottom
        grid_bottom_y = torch.abs(self.y).view(self.pano_height,self.pano_width,1).expand(-1,-1,3)
        grid_bottom_tmp = self.sphere_coor/grid_bottom_y
        grid_bottom_u = grid_bottom_tmp[:,:,0]
        grid_bottom_v = -grid_bottom_tmp[:,:,2] # convert to uv, because grid_sample's grid value range:[-1,1]. -1 correspoding left-top,1 corresponding right-down
        grid_bottom = torch.stack([grid_bottom_u,grid_bottom_v],dim=2)
        mask_bottom = ((grid_bottom_u)>=-1) * ((grid_bottom_u)<=1) * ((grid_bottom_v)>=-1) * ((grid_bottom_v)<=1) * ((grid_bottom_tmp[:,:,1]<0.0))
        mask_bottom = mask_bottom.unsqueeze(-1).float()

        return torch.stack([grid_left,grid_front,grid_right,grid_back,grid_top,grid_bottom],dim=0),\
            torch.stack([mask_left,mask_front,mask_right,mask_back,mask_top,mask_bottom],dim=0)

    def ToPano(self,input,mode='bilinear'):
        """
        cube 2 pano

        Args:
            input (torch.float32): shape: [b, c*6, h, w]
            mode (str, optional): interpolation mode. Defaults to 'bilinear'.

        Returns:
            torch.float32: shape:[b, c, h, w]
        """
        assert mode in ['bilinear', 'nearest', 'bicubic']
        image = input.reshape(6*self.batch_size,-1,(self.cube_lenth + 2 * self.cube_padding_size),(self.cube_lenth + 2 * self.cube_padding_size))
        mask_grid = self.grid * self.mask.expand_as(self.grid)
        mask_grid = (mask_grid+1)/2
        mask_grid = (mask_grid * self.cube_lenth + self.cube_padding_size) / (self.cube_lenth + 2 * self.cube_padding_size)*2-1.0
        
        output = F.grid_sample(image,mask_grid,mode=mode,padding_mode="border",align_corners=False)   #size:(6*batch_size,3,h,w)
        if self.is_cuda:
            final_mask = self.mask.permute(0,3,1,2).cuda()
        else:
            final_mask = self.mask.permute(0,3,1,2)
        output = output*final_mask
        output = output.unsqueeze(1).reshape(self.batch_size,6,-1,self.pano_height,self.pano_width)
        output = torch.sum(output,dim=1)    #size:(batch_size,3,h,w)
        return output

    def savePano(self,path,x):   # define batch_index=0
        fname,file_type = os.path.splitext(path)
        x = x.permute(0,2,3,1)
        for i in range(x.size()[0]):
            one = np.asarray(x[i].cpu(),dtype=np.float32)
            if file_type=='.exr' or file_type=='.hdr':
                cv2.imwrite(fname+str(i)+file_type,one)
            else:
                cv2.imwrite(fname+str(i)+file_type,one*255)
    
    def saveOnePano(self,path,x):   # define batch_index=0
        fname,file_type = os.path.splitext(path)
        x = x.permute(0,2,3,1)
        one = np.asarray(x[0].cpu(),dtype=np.float32)
        if file_type=='.exr' or file_type=='.hdr':
            cv2.imwrite(fname+file_type,one)
        else:
            cv2.imwrite(fname+file_type,one*255)




if __name__=="__main__":

    #-------------------------------------transfer cube to pano
    path = "/media/lz/3131-3739/HDR&Depth/CubemapData"
    target_path = "/media/lz/3131-3739/HDR&Depth/PanoData"

    items = os.listdir(path)
    if (len(items)%(6*2)!=0):
        exit(-1)
    num = round(len(items)/6/2)

    cube_lenth = 2048
    trans = Cube2Pano(batch_size=1,pano_height=int(2*cube_lenth),pano_width=int(4*cube_lenth),is_cuda=False,cube_lenth=cube_lenth,cube_padding_size=33)
    cube_list = [2,5,1,6,3,4]
    c_index = [2,1,0]

    # view convert
    # face order: 0-left,1-front,2-right,3-back,4-top,5-bottom
    horizon_angle = np.array([90.0,0.0,-90.0,-180.0,])/180.0*np.pi
    vertical_angle = np.array([90.0,-90.0])/180.0*np.pi
    rotate_list = []
    for  h in horizon_angle:
        temp = h*np.array([0,1,0],np.float32)      # rotate as y axis
        rotate_vector,_ = cv2.Rodrigues(temp)
        rotate_list.append(rotate_vector)

    for v in vertical_angle:
        temp = v*np.array([1,0,0],np.float32)      # rotate as x axis
        rotate_vector,_ = cv2.Rodrigues(temp)
        rotate_list.append(rotate_vector)
    rotate_list = [Variable(torch.FloatTensor(x)) for x in rotate_list]

    axis_x = torch.tensor([-1,0,0]).unsqueeze(-1).unsqueeze(-1).expand(3,cube_lenth,cube_lenth)  # size: (3, h, w)
    axis_y = torch.tensor([0,0,1]).unsqueeze(-1).unsqueeze(-1).expand(3,cube_lenth,cube_lenth)
    axis_z = torch.tensor([0,1,0]).unsqueeze(-1).unsqueeze(-1).expand(3,cube_lenth,cube_lenth)
    
    for ith in range(0,1):
        
        image_list = []
        for i in cube_list:
            img = cv2.imread(path+'/'+"CubeMap"+str(6*ith+i)+"_PreTonemapHDRColor.exr",cv2.IMREAD_UNCHANGED)[:,:,0:3]
            # img = img[29:2085,29:2085,:]
            
            h,w,c = img.shape
            img = np.asarray(img,dtype=np.float32)
            # img = img/255.0
            img = tranforms.ToTensor()(img)
            img = torch.reshape(img,(1,c,h,w))
            image_list.append(img)
        image = torch.cat(image_list,dim=1)
        image_pano = trans.ToPano(image,'bilinear')
        image_pano = image_pano[0,:,:,:].permute(1,2,0).numpy()
        cv2.imwrite(os.path.join(target_path,str(ith)+'_image.hdr'),image_pano)

        image_list = []
        for i in cube_list:
            img = cv2.imread(path+'/'+"CubeMap"+str(6*ith+i)+"_BaseColor.exr",cv2.IMREAD_UNCHANGED)[:,:,0:3]
            # img = img[29:2085,29:2085,:]
            h,w,c = img.shape
            img = np.asarray(img,dtype=np.float32)
            # img = img/255.0
            img = tranforms.ToTensor()(img)
            img = torch.reshape(img,(1,c,h,w))
            image_list.append(img)
        image = torch.cat(image_list,dim=1)
        image_pano = trans.ToPano(image,'bilinear')
        image_pano = image_pano[0,:,:,:].permute(1,2,0).numpy()
        cv2.imwrite(os.path.join(target_path,str(ith)+'_albedo.hdr'),image_pano)

        # image_list = []
        # for i in cube_list:
        #     img = cv2.imread(path+'/'+"CubeMap"+str(6*ith+i)+"_Metallic.exr",cv2.IMREAD_UNCHANGED)[:,:,0:3]
        #     h,w,c = img.shape
        #     img = np.asarray(img,dtype=np.float32)
        #     # img = img/255.0
        #     img = tranforms.ToTensor()(img)
        #     img = torch.reshape(img,(1,c,h,w))
        #     image_list.append(img)
        # image = torch.cat(image_list,dim=1)
        # image_pano = trans.ToPano(image)
        # image_pano = image_pano[0,:,:,:].permute(1,2,0).numpy()
        # image_pano = cv2.resize(image_pano, (4096,2048))
        # cv2.imwrite(os.path.join(target_path,str(ith)+'_metallic.hdr'),image_pano)

        # image_list = []
        # for i in cube_list:
        #     img = cv2.imread(path+'/'+"CubeMap"+str(6*ith+i)+"_Roughness.exr",cv2.IMREAD_UNCHANGED)[:,:,0:3]
        #     h,w,c = img.shape
        #     img = np.asarray(img,dtype=np.float32)
        #     # img = img/255.0
        #     img = tranforms.ToTensor()(img)
        #     img = torch.reshape(img,(1,c,h,w))
        #     image_list.append(img)
        # image = torch.cat(image_list,dim=1)
        # image_pano = trans.ToPano(image)
        # image_pano = image_pano[0,:,:,:].permute(1,2,0).numpy()
        # image_pano = cv2.resize(image_pano, (4096,2048))
        # cv2.imwrite(os.path.join(target_path,str(ith)+'_roughness.hdr'),image_pano)

        # image_list = []
        # for _,i in enumerate(cube_list):
        #     img = cv2.imread(path+'/'+"CubeMap"+str(6*ith+i)+"_WorldNormal.exr",cv2.IMREAD_UNCHANGED)[:,:,0:3]
        #     h,w,c = img.shape
        #     img = img[:,:,::-1].copy()# bgr->rgb
        #     img = np.asarray(img,dtype=np.float32)
        #     # img = img/255.0
        #     img = tranforms.ToTensor()(img)
            
        #     img = img*2.0-1.0
        #     img = img[0:1,:,:] * axis_x + img[1:2,:,:] * axis_y + img[2:3,:,:] * axis_z
        #     img = F.normalize(img,dim=0)

        #     # left hand to right hand axis
        #     img[2,:,:] = img[2,:,:] * -1
        #     # rotate normal from six face's local view to front view
        #     img = torch.einsum('ab,bcd->acd',(rotate_list[_],img))
        #     img = F.normalize(img,dim=0)    #value:[-1,1] normal vector
        #     img = (img+1.0)*0.5
        #     img = torch.reshape(img,(1,c,h,w))
        #     image_list.append(img)
        # image = torch.cat(image_list,dim=1)
        # image_pano = trans.ToPano(image)
        # image_pano = image_pano[0,c_index,:,:].permute(1,2,0).numpy()
        # image_pano = cv2.resize(image_pano, (4096,2048))
        # cv2.imwrite(os.path.join(target_path,str(ith)+'_normal.hdr'),image_pano)

        # image_list = []
        # for i in cube_list:
        #     img = cv2.imread(path+'/'+"CubeMap"+str(6*ith+i)+"_SceneDepth.exr",cv2.IMREAD_UNCHANGED)[:,:,0:3]
        #     img = img[33:2081,33:2081,:]
        #     h,w,c = img.shape
        #     img = np.asarray(img,dtype=np.float32)
        #     # img = img/255.0
        #     img = tranforms.ToTensor()(img)
        #     img = torch.reshape(img,(1,c,h,w))
        #     img = img/2000.0
        #     image_list.append(img)
        # image = torch.cat(image_list,dim=1)
        # image_pano = trans.ToPano(image)
        # image_pano = image_pano[0,:,:,:].permute(1,2,0).numpy()
        # cv2.imwrite(os.path.join(target_path,str(ith)+'_depth.hdr'),image_pano)

        image_list = []
        for i in cube_list:
            img = cv2.imread(path+'/'+"CubeMap"+str(6*ith+i)+"_PostTonemapHDRColor.exr",cv2.IMREAD_UNCHANGED)[:,:,0:3]
            # img = img[33:2081,33:2081,:]
            h,w,c = img.shape
            img = np.asarray(img,dtype=np.float32)
            # img = img/255.0
            img = tranforms.ToTensor()(img)
            img = torch.reshape(img,(1,c,h,w))
            image_list.append(img)
        image = torch.cat(image_list,dim=1)
        image_pano = trans.ToPano(image,'bilinear')
        image_pano = image_pano[0,:,:,:].permute(1,2,0).numpy()
        cv2.imwrite(os.path.join(target_path,str(ith)+'_image.png'),image_pano*255)


        