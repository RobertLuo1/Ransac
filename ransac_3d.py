"""
--coding:utf-8--
@author:58119327 罗卓彦
@email:920354187@qq.com
@title:RANSAC_3d & Least Square -> plane fitting
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

def generate_data(a,b,d,sample_size):
    """
    为了验证ransac的效果，将平面表示成z=ax+by+d的方式,而不是完全的随机生成数据，生
    成数据方式将固定x，y轴来生成z轴的点，并在其中适当添加噪声点
    :param a: x的系数
    :param b: y的系数
    :param d: 截距
    :param sample_size: 生成样本多少
    :return: 生成好的数据
    """
    random_x =[]#存放初始数据
    random_y = []#存放初始数据
    random_z = []#存放生成的数据
    X = np.linspace(0,10,sample_size)#规定在0-10中等比例生成
    Y = np.linspace(0,10,sample_size)#y也是，再规定的0-10中等比例生成
    Z = a*X+b*Y+d#这样是方便验证最终的结果，方便最后的调参工作
    for i in range(sample_size):
        random_x.append(X[i])#将生成的数据点加入到列表中
        random_y.append(Y[i])#将生成的数据点加入到列表中
        random_z.append(Z[i]+np.random.normal(0,1))#添加高斯噪声
    for i in range(sample_size):
        random_x.append(np.random.uniform(0,10))#在生成的点中添加一些噪声
        random_y.append(np.random.uniform(0,10))#在生成的点中添加一些噪声
        random_z.append(np.random.uniform(Z.min(),Z.max()))
    data = np.array([random_x,random_y,random_z]).T#最后把三维数据点集合起来
    np.random.shuffle(data)#随机打乱这些点来更好适应实际的状况
    size = sample_size*2#等数量添加了个噪声点
    return (data,size)

def distance(point_0,point_1,point_2,point_3):
    """
    根据随机找到的三个点来进行计算平面的一般式，再通过一般式来计算点到平面的距离以此来判断这个点是否为内点
    :param point_0: 待判断的点
    :param point_1: 随机选取的点
    :param point_2: 随机选取的点
    :param point_3: 随机选取的点
    :return: 距离distance
    """
    # (y3 - y1)*(z3 - z1) - (z2 -z1)*(y3 - y1);
    #(x3 - x1)*(z2 - z1) - (x2 - x1)*(z3 - z1)
    #(x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
    #-(A * x1 + B * y1 + C * z1)
    A = (point_3[1]-point_1[1])*(point_3[2]-point_1[2]) - (point_2[2]-point_1[2])*(point_3[1]-point_1[1])
    B = (point_3[0]-point_1[0])*(point_2[2]-point_1[2]) - (point_2[0]-point_1[0])*(point_3[2]-point_1[2])
    C = (point_2[0]-point_1[0])*(point_3[1]-point_1[1]) - (point_3[0]-point_1[0])*(point_2[1]-point_1[1])
    D = -(A*point_1[0]+B*point_1[1]+C*point_1[2])
    d = np.abs(A*point_0[0]+B*point_0[1]+C*point_0[2]+D)/np.sqrt(A**2+B**2+C**2)#通过公式得到这个点到这个平面的距离
    return d#返回最终的距离

def ransac(data,sample_size,iteration,threshold,probability):
    """
    这里是实现ransac，需要几个超参数，然后通过自适应的方式实现ransac，最终给出内点集合
    :param data: 待拟合的数据点
    :param sample_size: #样本总数
    :param iteration: 循环次数
    :param threshold: 门限，用于判断是否为内点
    :param probability: 概率，计算循环次数
    :return: 内点集合
    """
    inliers = []#内点集合
    temp = []#一个中间变量，用于存放内点
    total_inliers = 0#计数器
    before = 0#存放之前的内点数用于判断
    sample_count = 0
    while sample_count<iteration:
        temp.clear()#首先清空这一次临时的内点集合
        index_1,index_2,index_3 = np.random.choice(range(sample_size),3,replace=False)#随机挑选三个点
        point_1 = data[index_1]#得到第一个点
        point_2 = data[index_2]#得到第二个点
        point_3 = data[index_3]#得到第三个点
        total_inliers = 0
        for i in range(sample_size):
            d = distance(data[i],point_1,point_2,point_3)#获得当前点到这个平面的距离
            if d <= threshold:#如果距离小于门限
                temp.append(data[i])#将这个点视为内点
                total_inliers+=1
        if total_inliers > before:#如果此次内点数比之前的要多
            inliers = temp[:]#更换内点
            iteration = np.log(1 - probability) / np.log(1 - np.power((total_inliers / sample_size), 3))#重新计算循环次数
            before = total_inliers#更改此时的内点数
        sample_count += 1
    return (inliers)

def least_square(inliers,learning_rate,iters):
    """
    最小二乘法拟合平面，这里采用梯度下降的方式，之所以不采用闭式解的方式是有原因的
    :param inliers: 内点集合
    :param learning_rate: 学习率
    :param iters: 循环次数也可以是epoch
    :return: 最终平面的方程
    """
    xy = inliers[:,:2]
    z = inliers[:,2:]#做一个数据的改动，便于实现梯度下降法
    xy_augment = np.hstack([np.ones((xy.shape[0],1)),xy])#将其添加一项，是bias
    samples,featurs = xy_augment.shape#获得这个样本的数量和特征
    weights = np.ones((featurs,1))#初始化权重
    loss = []#计算loss，用于之后的调参工作
    # batch_size = 8
    # for i in range(iters):
    #     for j in range(0,samples,batch_size):
    #         z_predicted = xy_augment[j:j+batch_size].dot(weights)
    #         error = np.linalg.norm(z_predicted-z[j:j+batch_size])
    #         loss.append(error)
    #         dw = xy_augment[j:j+batch_size].T.dot(z_predicted-z[j:j+batch_size])
    #         weights -= (2/8)*learning_rate*dw
    for i in range(iters):#实现梯度下降
        z_predict = xy_augment.dot(weights)#计算predict
        error = np.linalg.norm(z_predict-z)#保存error易于调参
        loss.append(error)
        dw = xy_augment.T.dot(z_predict-z)#计算梯度
        weights -= (2/samples)*learning_rate*dw#更新权值
    # plt.plot(loss,label='loss')
    # plt.legend()
    # plt.show()
    # linear_model = LinearRegression()
    # linear_model.fit(xy_augment,z)
    # return (linear_model.coef_,linear_model.intercept_)
    # weights = np.linalg.inv(np.dot(xy_augment.T,xy_augment)).dot(xy_augment.T).dot(z)#normal Equation
    return np.squeeze(weights)#返回权值

np.random.seed(10)#设置随机种子，保证调参效果
data,sample_size = generate_data(2,3,10,100)#生成数据
fig = plt.figure()#画图
ax = plt.axes(projection='3d')#画三维图
ax.scatter3D(data[:,0],data[:,1],data[:,2],cmap='b')
ax.set(xlabel='X',
       ylabel='Y',
       zlabel='Z',)
ax.view_init(elev=45,azim=0)#调整图形的展示方向，保证能完整看出平面
iteration = 200#设置的参数，循环200次
threshold = 2.0#内点与平面的阈值设置为2.0都是超参数的设置
probability = 0.99
inliers = np.array(ransac(data,sample_size,iteration,threshold,probability))#得到相应的内点集合
weights = least_square(inliers,learning_rate=0.015,iters=700)#再对内点集合进行一个平面的拟合
# weights = least_square(inliers,learning_rate=0.001,iters=700)#再对内点集合进行一个平面的拟合
D = weights[0]
A = weights[1]
B = weights[2]
X = data[:,0]
Y = data[:,1]
X, Y = np.meshgrid(X, Y)
Z = (A*X+B*Y+D)
ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap='rainbow')
text = f"best_A ={A:.3f} \n"+f"best_B ={B:.3f} \n"+f"best_D = {D:.3f}"
ax.text(8,9,10, text,fontdict={'size': 8, 'color': 'r'})
plt.show()

#观看内点的分布，便于调参
# fig2 = plt.figure()#画图
# ax = plt.axes(projection='3d')#画三维图
# ax.scatter3D(inliers[:,0],inliers[:,1],inliers[:,2],cmap='b')##作内点图，观看效果
# ax.set(xlabel='X',
#        ylabel='Y',
#        zlabel='Z',)
# ax.view_init(elev=45,azim=0)#调整图形的展示方向，保证能完整看出平面
# plt.show()