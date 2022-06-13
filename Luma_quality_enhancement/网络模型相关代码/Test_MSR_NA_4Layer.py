from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import random
import math
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

from MSR_NA_4Layer import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
class YUV:
    def __init__(self, height, width, y, u, v):
        self.height = height
        self.width = width
        self.y = y
        self.u = u
        self.v = v
def YUVread(path, h, w):
    y_all = np.uint8([])
    u_all = np.uint8([])
    v_all = np.uint8([])
    with open(path, 'rb')as file:
        y = np.uint8(list(file.read(w * h)))
        u = np.uint8(list(file.read(w * h >> 2)))
        v = np.uint8(list(file.read(w * h >> 2)))
        y_all = np.concatenate([y_all, y])
        u_all = np.concatenate([u_all, u])
        v_all = np.concatenate([v_all, v])
        y_all = np.reshape(y_all, [1, h, w])
        u_all = np.reshape(u_all, [1, h >> 1, w >> 1])
        v_all = np.reshape(v_all, [1, h >> 1, w >> 1])
    return y_all, u_all, v_all
def cut_patch(video_in, patch_size, center_x_Y, center_y_Y, center_x_UV, center_y_UV):
    patch_Y = video_in.y[0, center_y_Y:center_y_Y + patch_size, center_x_Y:center_x_Y + patch_size]
    patch_Y = np.reshape(patch_Y, [1, patch_size, patch_size])
    patch_Y = torch.from_numpy(patch_Y)
    patch_Y = patch_Y.float()
    patch_uv = patch_size // 2
    patch_U = video_in.u[0, center_y_UV:center_y_UV + patch_uv, center_x_UV:center_x_UV + patch_uv]
    patch_U = np.reshape(patch_U, [1, patch_uv, patch_uv])
    patch_U = torch.from_numpy(patch_U)
    patch_U = patch_U.float()
    patch_V = video_in.v[0, center_y_UV:center_y_UV + patch_uv, center_x_UV:center_x_UV + patch_uv]
    patch_V = np.reshape(patch_V, [1, patch_uv, patch_uv])
    patch_V = torch.from_numpy(patch_V)
    patch_V = patch_V.float()
    return patch_Y, patch_U, patch_V
def YUVwrite(path, y, u, v):
    if type(y) is not np.ndarray:
        y = y.cpu()

        y = y.detach().numpy()
        y = y.astype(np.uint8)
    if type(u) is not np.ndarray:
        u = u.cpu()

        u = u.detach().numpy()
        u = u.astype(np.uint8)  ########必须是uint8才能正常显示
    if type(v) is not np.ndarray:
        v = v.cpu()

        v = v.detach().numpy()
        v = v.astype(np.uint8)

    with open(path, 'wb') as file:
        for fn in range(1):
            file.write(y.tobytes())
            file.write(u.tobytes())
            file.write(v.tobytes())
class Imagedata(Dataset):
    def __init__(self,img_path,label_path):
        self.img_path=img_path
        self.label_path=label_path
        imgs=os.listdir(img_path)
        labels=os.listdir(label_path)
        imgs=sorted(imgs,key=lambda x:int(os.path.splitext(x)[0][0:3]))
        labels=sorted(labels,key=lambda x:int(os.path.splitext(x)[0][0:3]))
        self.img_label=list(zip(imgs,labels))
    def __getitem__(self, index):
        x_path,y_path=self.img_label[index]
        w = 128
        h = 128
        ################################################
        start = x_path.rfind('_')
        end = x_path.rfind('.')
        theQP = int(x_path[start + 1:end])
        QP = torch.ones([1, 128, 128]) * theQP / 63.0
        total_x_path=self.img_path+x_path
        total_y_label=self.label_path+y_path
        patchsize=128
        y_input,u_input,v_input=YUVread(total_x_path,h,w)
        y_label,u_label,v_label=YUVread(total_y_label,h,w)
        input_video = YUV(h, w, y_input, u_input, v_input)
        label_video=YUV(h,w,y_label,u_label,v_label)
        #生成patch的左上角随机点
        center_y_Y=0
        center_x_Y=0
        center_y_UV=center_y_Y//2
        center_x_UV=center_x_Y//2
        #获取待预测块的和待预测块的真实值
        patch_y_input,patch_u_input,patch_v_input=cut_patch(input_video,patchsize,center_x_Y,center_y_Y,center_x_UV,center_y_UV)
        #YUVwrite('G:\\思路\\intra\\idea1_train\\参考图片\\' + str(index) + '_input.yuv', patch_y_input,patch_u_input,patch_v_input)
        patch_y_label,patch_u_label,patch_v_label=cut_patch(label_video, patchsize, center_x_Y,center_y_Y ,center_x_UV, center_y_UV)
        input_y= patch_y_input/255.0
        label_y=patch_y_label
        #label_uv=torch.from_numpy(label_uv)
        #label_uv=label_uv.float()

        return input_y,QP, label_y
    def __len__(self):
        return len(self.img_label)


def loadtraindata(inputpath,labelpath):
    trainset = Imagedata(inputpath, labelpath)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, drop_last=False,num_workers=8,pin_memory=True)
    return trainloader

def testmodel():
    # 加载多次网络模型，进行一个图片的多个块的预测，查看一个图片多个块的平均PSNR；
    # 使用待测图片的最小尺寸进行块的数量的选取；
    # ClassA:3840x2160,6
    # ClassB:1920x1080,7
    # ClassC:832x480,4
    # ClassE:1280x720,3
    #########
    pathroot="G:\\IDEA2\\CTC测试集关闭环路滤波器\\patch数据集\\ClassB\\"
    QPNum=8
    PictureNum=6
    psnrnumpy = np.zeros((QPNum, PictureNum))#存储每张图片的测试平均PSNR
    #########################################################
    criterion = nn.MSELoss()
    net = torch.load('G:\\idea2_netparam\\MSR_NA_4Layer_89net.pkl',
                     map_location=lambda storage, loc: storage)
    net = net.to(device)
    ################################################################
    file_names_dir = os.listdir(pathroot)
    for i in range(len(file_names_dir)):
        file_name_in_path = os.path.join(pathroot, file_names_dir[i])#进入到具体QP的路径
        thefile_names_dir = os.listdir(file_name_in_path)
        for j in range(len(thefile_names_dir)):
            mse=0
            thepath=os.path.join( file_name_in_path+'\\', thefile_names_dir [j])##到了某个QP下的某个具体帧
            trainloader = loadtraindata(thepath+"\\"+"input\\",thepath+"\\"+"label\\")
            len1 = os.listdir(thepath+"\\"+"input\\")
            numpatch=len(  len1)
            patcb_psnr=0
            bar = tqdm(trainloader)
            for input_y, QP, label_y in bar:
                input_y = input_y.to(device)
                QP = QP.to(device)
                label_y = label_y.to(device)
                with torch.no_grad():
                    output = net(input_y, QP) * 255.0
                    mse=criterion(output, label_y)
                    onepsnr=10 * math.log((255 ** 2) / mse, 10)
                    patcb_psnr=patcb_psnr+onepsnr
            n=int(numpatch/1)
            #mse=mse/n
            PSNR = patcb_psnr/n##到了某个QP下的某个具体帧的平均PSNR
            psnrnumpy[i][j]=PSNR
    psnrtrans=psnrnumpy.transpose()
    print(psnrtrans)
    np.savetxt('cuoClassB_MSR_NA_4Layer_89net.txt',  psnrtrans)

if __name__ == '__main__':
    testmodel()