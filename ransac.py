import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression


def generated_data(sample_size,weight,bias):
    X = np.linspace(0, 10, sample_size)
    Y = weight * X + bias
    random_x = []
    random_y = []
    for i in range(sample_size):
        random_x.append(X[i])
        random_y.append(Y[i] + np.random.normal(0, 1))
    for _ in range(sample_size):
        random_x.append(np.random.uniform(0,10))
        random_y.append(np.random.uniform(10,30))
    return (random_x,random_y,sample_size*2)


def distance(index,k,b):
    x_0 = random_x[index]
    y_0 = random_y[index]
    d = np.abs(k*x_0+b-y_0)/np.sqrt(1+k**2)
    return d

def ransac(sample_size,iteration,threshold,probability):
    inliers_x = []
    inliers_y = []
    inliers_x_temp = []
    inliers_y_temp = []
    before = 0
    sample_count = 0
    while sample_count<iteration:
        inliers_x_temp.clear()
        inliers_y_temp.clear()
        index_1,index_2 = np.random.choice(range(sample_size),2,replace=False)
        x_1 = random_x[index_1]
        y_1 = random_y[index_1]
        x_2 = random_x[index_2]
        y_2 = random_y[index_2]
        k = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - k * x_1
        for i in range(sample_size):
            d = distance(i,k,b)
            if d <= threshold:
                inliers_x_temp.append(random_x[i])
                inliers_y_temp.append(random_y[i])
        if len(inliers_x_temp) > before:
            inliers_x = inliers_x_temp[:]
            inliers_y = inliers_y_temp[:]
            before = len(inliers_x)
            iteration = np.log(1-probability)/np.log(1-np.power(len(inliers_x)/sample_size,2))
        if len(inliers_x) >=sample_size/2:
            return (inliers_x,inliers_y)
        sample_count+=1
    return (inliers_x,inliers_y)

def least_square(sample_size,iteration,threshold,probability):
    X,Y = ransac(sample_size,iteration,threshold,probability)
    X = np.array(X).reshape(-1,1)
    X = np.hstack([X,np.ones_like(X)])
    Y = np.array(Y).reshape(-1,1)
    # linear_model = LinearRegression()
    # linear_model.fit(X,Y)
    # k =  linear_model.coef_[0][0]
    # m =  linear_model.intercept_[0]
    M = np.squeeze(np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(Y))
    k = M[0]
    m = M[1]
    # U = np.concatenate((X-X.mean(),Y-Y.mean()),axis=1)
    # eigvalue, eigvector = np.linalg.eig(np.dot(U.T,U))
    # min_value = np.min(eigvalue)
    # N = eigvector[np.where(eigvalue==min_value)][0]
    # d = N[0]*X.mean()+N[1]*Y.mean()
    # return (N[0],N[1],d)
    return (k,m)


np.random.seed(10)
threshold = 0.6
probability = 0.99
iteration = 200

random_x, random_y,sample_size = generated_data(100,2,10)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_title('Ransac')
ax1.scatter(random_x, random_y)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
k,m = least_square(sample_size,iteration,threshold,probability)
# k,m = total_least_square(sample_size,iteration,threshold,probability)
Y = k*np.array(random_x)+m
ax1.plot(random_x, Y)
text = f"best_k ={k:.3f} \n"+f"best_m ={m:.3f}"
plt.text(8,9, text,fontdict={'size': 8, 'color': 'r'})
plt.show()

# inliers_x,inlier_y = ransac(sample_size,iteration,threshold,probability)
# plt.scatter(inliers_x,inlier_y)
# plt.show()
# plt.scatter(random_x,random_y)
# plt.title('ransac')
# plt.show()











