# -*- coding: utf-8 -*-
#想要加中文注释就必须将字符编码格式设置为utf8
#作者:郭开

import numpy as np
import os
import cv2
import cv2.cv as cv
from skimage import transform as tf
from PIL import Image, ImageDraw
import threading
from time import ctime,sleep
import time
import sklearn
import matplotlib.pyplot as plt
import skimage

caffe_root = '/home/gk/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import sklearn.metrics.pairwise as pw

#保存人脸的位置
global face_rect
face_rect=[]
#我把GPU加速注释掉了,所以没有GPU加速,速度有点慢,你要在学校有条件找个有GeForce显卡的电脑
#caffe.set_mode_gpu()

#加载caffe模型
global net
net=caffe.Classifier('/home/gk/caffe-master/examples/VGG_face/vgg_face_caffe/VGG_FACE_deploy.prototxt',
    '/home/gk/caffe-master/examples/VGG_face/vgg_face_caffe/VGG_FACE.caffemodel')


def detect(img, cascade):
    rects = cv.HaarDetectObjects(img, cascade, cv.CreateMemStorage(), 1.1, 2,cv.CV_HAAR_DO_CANNY_PRUNING, (255,255))#CV_HAAR_SCALE_IMAGE，按比例正常检测
    if len(rects) == 0:
        return []
    result = []
    #将检测到的位置保存到result
    for r in rects:
        result.append((r[0][0], r[0][1], r[0][0]+r[0][2], r[0][1]+r[0][3]))
    #返回人脸的位置和大小,大小限定在300~500之间
    if result[0][2]> 300 and result[0][3] > 300 and result[0][2]< 500 and result[0][3] < 500:
        return result
    else:
        return []

#画绿色的人脸框
def draw_rects(img, rects, color):
    if rects:
        for i in rects:
            cv.Rectangle(img, (int(rects[0][0]), int(rects[0][1])),(int(rects[0][2]),int(rects[0][3])),cv.CV_RGB(0, 255, 0), 1, 8, 0)#画一个绿色的矩形框

#用来注册一个用户
def register(path,img,rects):
    if rects:
        #保证图片是N*N的,即正方形
            if rects[0][2]<rects[0][3]:
                cv.SetImageROI(img,(rects[0][0]+10, rects[0][1]+10,rects[0][2]-50,rects[0][2]-50))
            else:
                cv.SetImageROI(img,(rects[0][0]+10, rects[0][1]+10,rects[0][3]-50,rects[0][3]-50))
            dst=cv.CreateImage((224,224), 8, 3)
        #保存人脸
            cv.Resize(img,dst,cv.CV_INTER_LINEAR)
            cv.SaveImage(path,dst)

#用来识别一个用户
def recog(md,img):
    global face_rect
    src_path='./regist_pic/'+str(md)
    while True:
        rects=face_rect
        if rects:
            #img保存用来验证的人脸
            if rects[0][2]<rects[0][3]:
                cv.SetImageROI(img,(rects[0][0]+10, rects[0][1]+10,rects[0][2]-100,rects[0][2]-100))
            else:
                cv.SetImageROI(img,(rects[0][0]+10, rects[0][1]+10,rects[0][3]-100,rects[0][3]-100))
            #将img暂时保存起来
            dst=cv.CreateImage((224,224), 8, 3)
            cv.Resize(img,dst,cv.CV_INTER_LINEAR)
            cv.SaveImage('./temp.bmp',dst)
            #取出5张注册的人脸,分别与带验证的人脸进行匹配,可以得到五个相似度,保存到scores中
            scores=[]
            for i in range(5):
                res=compar_pic('./temp.bmp',src_path+'/'+str(i)+'.bmp')
                scores.append(res)
                print res
            #求scores的均值
            result=avg(scores)
            print 'avg is :',avg(scores)
            return result


def avg(scores):
    max=scores[0]
    min=scores[0]
    res=0.0
    for i in scores:
        res=res+i
        if min>i:
            min=i
        if max<i:
            max=i
    return (res-min-max)/3


def compar_pic(path1,path2):
    global net
    #加载验证图片
    X=read_image(path1)
    test_num=np.shape(X)[0]
    #X  作为 模型的输入
    out = net.forward_all(data = X)
    #fc7是模型的输出,也就是特征值
    feature1 = np.float64(out['fc7'])
    feature1=np.reshape(feature1,(test_num,4096))
    #np.savetxt('feature1.txt', feature1, delimiter=',')

    #加载注册图片
    X=read_image(path2)
    #X  作为 模型的输入
    out = net.forward_all(data=X)
    #fc7是模型的输出,也就是特征值
    feature2 = np.float64(out['fc7'])
    feature2=np.reshape(feature2,(test_num,4096))
    #np.savetxt('feature2.txt', feature2, delimiter=',')
    #求两个特征向量的cos值,并作为是否相似的依据
    predicts=pw.cosine_similarity(feature1, feature2)
    return  predicts



def read_image(filelist):

    averageImg = [129.1863,104.7624,93.5940]
    X=np.empty((1,3,224,224))
    word=filelist.split('\n')
    filename=word[0]
    im1=skimage.io.imread(filename,as_grey=False)
    image =skimage.transform.resize(im1,(224, 224))*255
    X[0,0,:,:]=image[:,:,0]-averageImg[0]
    X[0,1,:,:]=image[:,:,1]-averageImg[1]
    X[0,2,:,:]=image[:,:,2]-averageImg[2]
    return X


#用来显示当前图片
#Opencv中人脸检测的一个级联分类器
cascade = cv.Load("./haarcascade_frontalface_alt.xml")
#获取视频流的接口，0表示摄像头的id号，当只连接一个摄像头时默认为0
cam = cv.CaptureFromCAM(0)


def show_img():
    global face_rect
    #一个死循环，用来不间断的显示图片
    while True:
        img = cv.QueryFrame(cam)# 取出视频中的一帧
        #保存三通道的图片
        src=cv.CreateImage((img.width, img.height), 8, 3)
        cv.Resize(img,src,cv.CV_INTER_LINEAR)
        #保存灰度图片
        gray=cv.CreateImage((img.width, img.height), 8, 1)
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)#将rgb图片变成灰度图
        cv.EqualizeHist(gray,gray)#对灰度图进行直方图均衡化
        rects = detect(gray, cascade)#传入图片和分类器，如果检测到人脸，返回人脸的坐标和大小
        face_rect=rects
        #话那个绿色的人脸框
        draw_rects(src, rects, (0, 255, 0))

        #显示画框的人脸
        cv.ShowImage('DeepFace Wang_jun_qian', src)
        cv2.waitKey(5) == 27
    cv2.destroyAllWindows()



t1 = threading.Thread(target=show_img)


if __name__ == '__main__':
    #face_rect,用来保存人脸的位置,全局共享
    global face_rect
    #用来显示摄像头和画人脸框
    t1.start()
    while True:
        pattern=raw_input('注册输入1\n识别输入2\n请选择程序模式:')
        if pattern=='1':
            tag=0
            reg_id=raw_input('请输入注册id:')
            reg_path='./regist_pic'
            #判断用户是否已经注册
            dir_rec=os.listdir(reg_path)
            for subdir in dir_rec:
                if(subdir==reg_id):#说明该用户已经注册
                    print '该用户已经注册!!!!!!\n'
                    tag=1
            #该用户未注册
            if tag==0:
                #生成该用户的文件夹和注册图片
                os.mkdir(reg_path+'/'+reg_id)
                num=-2
                #注册五张人脸
                while num<4:
                    if face_rect:
                        num=num+1
                        if num>=0:
                            register_path=reg_path+'/'+str(reg_id)+'/'+str(num)+'.bmp'
                            register(register_path,cv.QueryFrame(cam),face_rect)
                            print 'now is '+str(num)+'........\n'
                            time.sleep(0.5)



        if pattern=='2':
            #md保存验证的id
            md=raw_input('请输入识别id:')
            #判断该用户是否存在
            tag=0
            dir_rec=os.listdir('./regist_pic')
            for subdir in dir_rec:
                if(subdir==md):#说明该用户存在
                    tag=1
            if tag==1:
                #阈值,大于这个值说明两个类的距离较远,不是一类
                thershold=0.85
                #把捕捉到的图片与注册的图片比较
                result=recog(md,cv.QueryFrame(cam))
                if result>=thershold:
                    print result
                    print '验证成功!!!!\n\n'
                else:
                    print '验证失败,不是本人!!!!\n\n'
            else:
                print '该用户不存在!!!!\n\n'


