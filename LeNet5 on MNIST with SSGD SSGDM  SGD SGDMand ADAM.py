import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import gzip
#import request
from abc import ABCMeta, abstractmethod
import copy
import sys
import os
print(sys.version)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    # 输入数据的形状
    # N：批数目，C：通道数，H：输入数据高，W：输入数据长
    N, C, H, W = input_data.shape  
    out_h = (H + 2*pad - filter_h)//stride + 1  # 输出数据的高
    out_w = (W + 2*pad - filter_w)//stride + 1  # 输出数据的长
    # 填充 H,W
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # (N, C, filter_h, filter_w, out_h, out_w)的0矩阵
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

# 设置当前路径
os.chdir(os.path.split(os.path.realpath(__file__))[0])

filename = [
	["training_images","train-images-idx3-ubyte.gz"],
	["test_images","t10k-images-idx3-ubyte.gz"],
	["training_labels","train-labels-idx1-ubyte.gz"],
	["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()
def draw_losses1(losses,losses1):
    label=[]
    t = np.arange(len(losses))
    _label = "sgd"
    _label1 = "dsgd"
    label.append(_label)
    label.append(_label1)
    plt.xlabel("迭代次数")
    plt.ylabel("测试精度")
    plt.grid(True)
    plt.legend(labels=label, loc="best")
    plt.plot(t, losses,'b-')
    plt.plot(t, losses1,'r:')
    plt.savefig("./测试精度对比图.jpg", bbox_inches='tight')
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

def Init(self,Train_Data,Train_Label):
        """
        这是初始化训练数据集的函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练标签集
        :return:
        """
        self.Train_Data = Train_Data            # 训练数据集
        self.Train_Label = Train_Label          # 训练数据集标签
        self.Size = np.shape(Train_Data)[0]     # 训练数据集规模性

def Shuffle_Sequence(size):
    """
    这是在运行SGD算法或者MBGD算法之前，随机打乱后原始数据集的函数
    :param size: 数据集规模
    """
    # 首先按照数据集规模生成自然数序列
    random_sequence = list(range(size))
    # 利用numpy的随机打乱函数打乱训练数据下标
    random_sequence = np.random.permutation(random_sequence)
    return random_sequence          # 返回数据集随机打乱后的数据序列
class FC():
    """
    Fully connected layer
    """
    def __init__(self, D_in, D_out):
        #print("Build FC")
        self.cache = None
        #self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in,D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}

    def _forward(self, X):
        # print("FC: _forward")
        # print(X.shape)
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def _backward(self, dout):
        # print("FC: _backward")
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
        self.b['grad'] = np.sum(dout, axis=0)
        #self._update_params()
        return dX

    def _update_params(self, lr=0.001):
        # Update the parameters
        self.W['val'] -= lr*self.W['grad']
        self.b['val'] -= lr*self.b['grad']

class ReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def _forward(self, X):
        #print("ReLU: _forward")
        out = np.maximum(0, X)
        self.cache = X
        return out

    def _backward(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX


class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        # print("Build Softmax")
        self.cache = None

    def _forward(self, X):
        # print("Softmax: _forward")
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def _backward(self, dout):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(dout,dZ)
        dX = np.dot(dX,dY)
        return dX



class Conv:
    	# 初始化权重（卷积核4维）、偏置、步幅、填充
    def __init__(self, Cin, Cout, X_H, X_W, F, stride, pad):

        self.stride = stride
        self.pad = pad
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization
        self.b = {'val': np.random.randn(Cout), 'grad': 0}
        self.X_H = X_H
        self.X_W = X_W
        self.Cin = Cin

    def _forward(self, x):
        x = x.reshape(x.shape[0], self.Cin, self.X_H, self.X_W)
        # 卷积核大小
        FN, C, FH, FW = self.W['val'].shape
        # 数据数据大小
        N, C, H, W = x.shape
        # 计算输出数据大小
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        # 利用im2col转换为行
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 卷积核转换为列，展开为2维数组
        col_W = self.W['val'].reshape(FN, -1).T
        # 计算正向传播
        out = np.dot(col, col_W) + self.b['val']
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def _backward(self, dout):
        # 卷积核大小
        FN, C, FH, FW = self.W['val'].shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        db = np.sum(dout, axis=0)
        dW = np.dot(self.col.T, dout)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)
   
        self.W['grad'] = dW
        self.b['grad'] = db

        dcol = np.dot(dout, self.col_W.T)
        # 逆转换
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class MaxPool:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def _forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
		# 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
		# 最大值
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def _backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred*Y_true, axis=1)
    for e in M:
        #print(e)
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss/N

class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax._forward(Y_pred)
        loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout



class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


class LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv(1, 6, 28, 28,5, 1, 2)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2,2)
        self.conv2 = Conv(6, 16, 14, 14, 5, 1, 0)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2,2)
        self.FC1 = FC(16*5*5, 120)
        self.ReLU3 = ReLU()
        self.FC2 = FC(120, 84)
        self.ReLU4 = ReLU()
        self.FC3 = FC(84, 10)
        self.Softmax = Softmax()
        self.p2_shape = None
        
    def forward(self, X):
        # print(X.shape)
        h1 = self.conv1._forward(X)
        a1 = self.ReLU1._forward(h1)
        p1 = self.pool1._forward(a1)
        h2 = self.conv2._forward(p1)
        a2 = self.ReLU2._forward(h2)
        p2 = self.pool2._forward(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0],-1) # Flatten 转化为列向量
        h3 = self.FC1._forward(fl)
        a3 = self.ReLU3._forward(h3)
        h4 = self.FC2._forward(a3)
        a5 = self.ReLU4._forward(h4)
        h5 = self.FC3._forward(a5)
        #a5 = self.Softmax._forward(h5)
        return h5

    def backward(self, dout):
        #dout = self.Softmax._backward(dout)
        dout = self.FC3._backward(dout)
        dout = self.ReLU4._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU3._backward(dout)
        dout = self.FC1._backward(dout)
        dout = dout.reshape(self.p2_shape) # reshape
        dout = self.pool2._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        
        dout = self.conv1._backward(dout)
        
    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params

class SGD(): 
    def __init__(self, params, lr=0.001, reg=0):
        self.parameters = params
        self.lr = lr
        self.reg = reg
    def step(self):        
        for param in self.parameters:
            param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])
 
class SGDMomentum():
    def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
        self.l = len(params)
        self.parameters = params
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
        self.lr = lr
        self.rho = momentum
        self.reg = reg

    def step(self):
        for i in range(self.l):
            self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
            self.parameters[i]['val'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['val'])


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.l = len(params)
        self.parameters = params
        self.moumentum = []
        self.velocities = []
        self.m_cat = []
        self.v_cat = []
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
            self.moumentum.append(np.zeros(param['val'].shape))
            self.v_cat.append(np.zeros(param['val'].shape))
            self.m_cat.append(np.zeros(param['val'].shape))

    
    def step(self):
        self.t += 1
        for i in range(self.l):
            g = self.parameters[i]['grad']
            self.moumentum[i]  = self.beta1 * self.moumentum[i]  + (1 - self.beta1) * g
            self.velocities[i] = self.beta2 * self.velocities[i] + (1 - self.beta2) * g * g
            self.m_cat[i] = self.moumentum[i]  / (1 - self.beta1 ** self.t)
            self.v_cat[i] = self.velocities[i] / (1 - self.beta2 ** self.t) 
            self.parameters[i]['val'] -= self.lr * self.m_cat[i] / (self.v_cat[i] ** 0.5 + self.epislon)
            



def normalize(e):
        s = []  
        for param in e:
            if param['grad'].ndim ==1:
                for i in range(len(param['grad'])):
                    param['grad'][i]=param['grad'][i]/(np.sqrt(s[i])+0.0000000001)
                s=[]
            
            if param['grad'].ndim ==4:
                m,n,k,l=param['grad'].shape
                
                for i in range(m):
                    s2 = 0
                    for j in range(n):
                        for h in range(k):
                            for o in range(l):
                                s2=s2+param['grad'][i,j,h,o]**2
                    s.append(s2)
                    param['grad'][i]=param['grad'][i]/(np.sqrt(s2)+0.0000000001)
            if param['grad'].T.ndim ==2:
                m,n=param['grad'].T.shape
                for i in range(m):
                    s2 = 0
                    for j in range(n):
                       s2=s2+param['grad'].T[i,j]**2
                    s.append(s2)
                    param['grad'].T[i]=param['grad'].T[i]/(np.sqrt(s2)+0.0000000001)
        return e
def uniondict(a,b):   
    if len(a):
        for i in range(len(a)):
            a[i]['grad']=a[i]['grad']+b[i]['grad']           
    else:
        a = copy.deepcopy(b)  
    return a
def lap(eps, g):

    for param in g:
        f = 0
        if param['grad'].ndim == 1:
            f = np.max(param['grad']) - np.min(param['grad'])
            for i in range(len(param['grad'])):
                param['grad'][i] = param['grad'][i] +np.random.laplace(0, f / eps)
       
        if param['grad'].ndim == 4:
            f = np.max(param['grad']) - np.min(param['grad'])
            m, n, k, l = param['grad'].shape
            for i in range(m):
                for j in range(n):
                    for h in range(k):
                        for o in range(l):
                            param['grad'][i, j, h, o]  = np.random.laplace(0, f / eps) + param['grad'][i, j, h, o]
        if param['grad'].T.ndim == 2:
            f = np.max(param['grad']) - np.min(param['grad'])
            m, n = param['grad'].T.shape
            for i in range(m):
                for j in range(n):
                    param['grad'].T[i, j] = np.random.laplace(0, f / eps) + param['grad'].T[i, j]
    return g





"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""            
##############################################################################
#Sgd, sgdm, adam, ssgd and ssgdm algorithm performance comparison
##############################################################################
'''
print("model = lenet-5")
model = LeNet5()
modelsgdm = LeNet5()
modeladam = LeNet5()
modelssgd = LeNet5()
modelssgdm = LeNet5()
batch_size =16
D_in = 784
D_out = 10
#print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))
size =16
#mnist.init()
X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

losses = []
losses2 = []
optim = SGD(model.get_params(), lr=0.0001, reg=0)
optimsgdm = SGDMomentum(modelsgdm.get_params(), lr=0.0005, momentum=0.999, reg=0.00000)
optimadam = Adam(modeladam.get_params(), lr=0.005, beta1=0.9, beta2=0.999, epislon=1e-8)
criterion = CrossEntropyLoss()
criterion1 = CrossEntropyLoss()
criterionsgdm = CrossEntropyLoss()
criterionadam = CrossEntropyLoss()
criterionssgdm = CrossEntropyLoss()
wr =modelssgd.get_params()
wr1 =modelssgdm.get_params()

ITER = 100
num=10
train_avg_accs = []
test_avg_accs = []
train_avg_accDs = []
test_avg_accDs = []
train_avg_accsgdms = []
test_avg_accsgdms = []
train_avg_accadams = []
test_avg_accadams = []
train_avg_accssgdms = []
test_avg_accssgdms = []
#sgd
for j in range(ITER):
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size*size )
    Y_batch = MakeOneHot(Y_batch, D_out)
    # forward, loss, backward, step
    Y_pred = model.forward(X_batch)
    loss, dout = criterion.get(Y_pred, Y_batch)
    model.backward(dout)
    optim.step()
    if j % num == 0:
        print("iter: %s" % (j))
        # save params
        weights = model.get_params()
        with open("weights.pkl", "wb") as f:
            pickle.dump(weights, f)

        test_size = 1000
        train_avg_acc = []
        test_avg_acc = []

        for i in range(10):
            #print("test_size = " + str(test_size))
            X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
            X_test_min, Y_test_min = get_batch(X_test, Y_test, test_size)
            # TRAIN SET ACC
            Y_pred_min = model.forward(X_train_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_train_min
            result = list(result)
            #print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train_min.shape[0]) + ", acc=" + str(result.count(0) / X_train_min.shape[0]))
            train_avg_acc.append(result.count(0) / X_train_min.shape[0])
            # TEST SET ACC
            Y_pred_min = model.forward(X_test_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_test_min
            result = list(result)
            #print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test_min.shape[0]) + ", acc=" + str(result.count(0) / X_test_min.shape[0]))
            test_avg_acc.append(result.count(0) / X_test_min.shape[0])
        print('sgd:' + '平均训练精度为', np.average(train_avg_acc))
        print('sgd:' + '平均测试精度为', np.average(test_avg_acc))
        train_avg_accs.append(np.average(train_avg_acc))
        test_avg_accs.append(np.average(test_avg_acc))
#sgdm
for j in range(ITER):
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size*size )
    Y_batch = MakeOneHot(Y_batch, D_out)
    # forward, loss, backward, step
    Y_pred = modelsgdm.forward(X_batch)
    loss, dout = criterionsgdm.get(Y_pred, Y_batch)
    modelsgdm.backward(dout)
    optimsgdm.step()
    if j % num == 0:
        print("iter: %s" % (j))
        # save params
        weightssgdm = modelsgdm.get_params()
        with open("weightssgdm.pkl", "wb") as f:
            pickle.dump(weightssgdm, f)
        test_size = 1000
        train_avg_accsgdm = []
        test_avg_accsgdm = []
        for i in range(10):
            #print("test_size = " + str(test_size))
            X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
            X_test_min, Y_test_min = get_batch(X_test, Y_test, test_size)

            # TRAIN SET ACC
            Y_pred_min = modelsgdm.forward(X_train_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_train_min
            result = list(result)
            #print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train_min.shape[0]) + ", acc=" + str(result.count(0) / X_train_min.shape[0]))
            train_avg_accsgdm.append(result.count(0) / X_train_min.shape[0])
            # TEST SET ACC
            Y_pred_min = modelsgdm.forward(X_test_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_test_min
            result = list(result)
            #print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test_min.shape[0]) + ", acc=" + str(result.count(0) / X_test_min.shape[0]))
            test_avg_accsgdm.append(result.count(0) / X_test_min.shape[0])
        print('sgdm:' + '平均训练精度为', np.average(train_avg_accsgdm))
        print('sgdm:' + '平均测试精度为', np.average(test_avg_accsgdm))
        train_avg_accsgdms.append(np.average(train_avg_accsgdm))
        test_avg_accsgdms.append(np.average(test_avg_accsgdm))
#adam
for j in range(ITER):
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size*size )
    Y_batch = MakeOneHot(Y_batch, D_out)
    # forward, loss, backward, step
    Y_pred = modeladam.forward(X_batch)
    loss, dout = criterionadam.get(Y_pred, Y_batch)
    modeladam.backward(dout)
    optimadam.step()
    if j % num == 0:
        print("iter: %s" % (j))
        # save params
        weightsadam = modeladam.get_params()
        with open("weightsadam.pkl", "wb") as f:
            pickle.dump(weightsadam, f)
        test_size = 1000
        train_avg_accadam = []
        test_avg_accadam = []
        for i in range(10):
            #print("test_size = " + str(test_size))
            X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
            X_test_min, Y_test_min = get_batch(X_test, Y_test, test_size)

            # TRAIN SET ACC
            Y_pred_min = modeladam.forward(X_train_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_train_min
            result = list(result)
            #print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train_min.shape[0]) + ", acc=" + str(result.count(0) / X_train_min.shape[0]))
            train_avg_accadam.append(result.count(0) / X_train_min.shape[0])
            # TEST SET ACC
            Y_pred_min = modeladam.forward(X_test_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_test_min
            result = list(result)
            #print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test_min.shape[0]) + ", acc=" + str(result.count(0) / X_test_min.shape[0]))
            test_avg_accadam.append(result.count(0) / X_test_min.shape[0])
        print('adam:' + '平均训练精度为', np.average(train_avg_accadam))
        print('adam:' + '平均测试精度为', np.average(test_avg_accadam))
        train_avg_accadams.append(np.average(train_avg_accadam))
        test_avg_accadams.append(np.average(test_avg_accadam))

#ssgd
for j in range(ITER):
    losses1 = []
    ws =[]
    for i in range(size):
        modelssgd.set_params(wr)
        X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
        Y_batch = MakeOneHot(Y_batch, D_out)
        #print('xunl',X_batch.shape)
        # forward, loss, backward, step
        Y_pred = modelssgd.forward(X_batch)
        loss, dout = criterion1 .get(Y_pred, Y_batch)
        modelssgd.backward(dout)
        w = modelssgd.get_params()

        ws=uniondict(ws, normalize(w))
    SGD(ws, lr=0.2/size, reg=0).step() 
    wr=copy.deepcopy(ws) 
    if j % num == 0:
        print("iter: %s" % (j))
        weights1 = modelssgd.get_params()
        with open("weights1.pkl","wb") as f:
            pickle.dump(weights1, f)
        test_size = 1000
        train_avg_accD = []
        test_avg_accD = []
        #print("test_size = " + str(test_size))
        for i in range(10):
            X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
            X_test_min,  Y_test_min  = get_batch(X_test,  Y_test,  test_size)

            # TRAIN SET ACC
            Y_pred_min = modelssgd.forward(X_train_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_train_min
            result = list(result)
            #print("TRAIN1--> Correct: " + str(result.count(0)) + " out of " + str(X_train_min.shape[0]) + ", acc=" + str(result.count(0)/X_train_min.shape[0]))
            train_avg_accD.append(result.count(0)/X_train_min.shape[0])
            # TEST SET ACC
            Y_pred_min = modelssgd.forward(X_test_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_test_min
            result = list(result)
            #print("TEST1--> Correct: " + str(result.count(0)) + " out of " + str(X_test_min.shape[0]) + ", acc=" + str(result.count(0)/X_test_min.shape[0]))
            test_avg_accD.append(result.count(0)/X_test_min.shape[0])
        print('ssgd:'+'平均训练精度为',np.average(train_avg_accD))
        print('ssgd:'+'平均测试精度为',np.average(test_avg_accD))
        train_avg_accDs.append(np.average(train_avg_accD))
        test_avg_accDs.append(np.average(test_avg_accD))
        # get batch, make onehot
#ssgdm
for j in range(ITER):
    losses1 = []
    ws1 =[]
    for i in range(size):
        modelssgdm.set_params(wr1)
        #print('model.get_params()',model.get_params()[1]['grad'])
        X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
        Y_batch = MakeOneHot(Y_batch, D_out)
        #print('xunl',X_batch.shape)
        # forward, loss, backward, step
        Y_pred = modelssgdm.forward(X_batch)
        loss, dout = criterionssgdm .get(Y_pred, Y_batch)
        modelssgdm.backward(dout)
        w1 = modelssgdm.get_params()
        #w1=copy.deepcopy(w)
        #ws=uniondict(ws, w)
        ws1=uniondict(ws1, normalize(w1))
    SGDMomentum(ws1, lr=20/(size*1.0002**j), momentum=0.99, reg=0.00000).step()
    wr1=copy.deepcopy(ws1) 
    #losses1.append(loss)
    if j % num == 0:
        print("iter: %s" % (j))
        weightsssgdm = modelssgdm.get_params()
        with open("weightsssgdm.pkl","wb") as f:
            pickle.dump(weightsssgdm, f)
        test_size = 1000
        train_avg_accssgdm = []
        test_avg_accssgdm = []
        #print("test_size = " + str(test_size))
        for i in range(10):
            X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
            X_test_min,  Y_test_min  = get_batch(X_test,  Y_test,  test_size)

            # TRAIN SET ACC
            Y_pred_min = modelssgdm.forward(X_train_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_train_min
            result = list(result)
            #print("TRAIN1--> Correct: " + str(result.count(0)) + " out of " + str(X_train_min.shape[0]) + ", acc=" + str(result.count(0)/X_train_min.shape[0]))
            train_avg_accssgdm.append(result.count(0)/X_train_min.shape[0])
            # TEST SET ACC
            Y_pred_min = modelssgdm.forward(X_test_min)
            result = np.argmax(Y_pred_min, axis=1) - Y_test_min
            result = list(result)
            #print("TEST1--> Correct: " + str(result.count(0)) + " out of " + str(X_test_min.shape[0]) + ", acc=" + str(result.count(0)/X_test_min.shape[0]))
            test_avg_accssgdm.append(result.count(0)/X_test_min.shape[0])
        print('ssgdm:'+'平均训练精度为',np.average(train_avg_accssgdm))
        print('ssgdm:'+'平均测试精度为',np.average(test_avg_accssgdm))
        train_avg_accssgdms.append(np.average(train_avg_accssgdm))
        test_avg_accssgdms.append(np.average(test_avg_accssgdm))


t = np.arange(len(train_avg_accssgdms))
plt.xlabel("num")
plt.ylabel("train_acc")
plt.title("train")
plt.grid(True)
plt.plot(t, train_avg_accs,'b-',label="sgd")
plt.plot(t, train_avg_accDs,'r:',label="ssgd")
plt.plot(t, train_avg_accsgdms,'p--',label="sgdm")
plt.plot(t, train_avg_accadams,'k-.',label="adam")
plt.plot(t, train_avg_accssgdms,'g',label="ssgdm")
plt.legend(loc='lower right')
plt.savefig("./训练精度对比图.jpg", bbox_inches='tight')
plt.show()
t = np.arange(len(test_avg_accssgdms))
plt.xlabel("num")
plt.ylabel("test_acc")
plt.title("test")
plt.grid(True)
plt.plot(t, test_avg_accs,'b-',label="sgd")
plt.plot(t, test_avg_accDs,'r:',label="ssbgd")
plt.plot(t, test_avg_accsgdms,'p-',label="sgdm")
plt.plot(t, test_avg_accadams,'k-.',label="adam")
plt.plot(t, test_avg_accssgdms,'g',label="ssgdm")
plt.legend(loc='lower right')
plt.savefig("./测试精度对比图.jpg", bbox_inches='tight')
plt.show()

'''

##############################################################################
#This code is a test of the robustness of the ssgd algorithm(ssgd with noise)
##############################################################################

print("model = lenet-5")
model1 = LeNet5()
train_avg_acc =[]
test_avg_acc =[]
D_in = 784
D_out = 10
#print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))
#mnist.init()
X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)
losses = []
#optim = SGD(model.get_params(), lr=0.0005, reg=0)
#optim = SGDMomentum(model.get_params(), lr=0.0001, momentum=0.999, reg=0.00000)
#optim = Adam(model.get_params(), lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8)
criterion = CrossEntropyLoss()
criterion1 = CrossEntropyLoss()
wr =model1.get_params()
ITER = 1000
batch_size =16
size =4
for j in range(ITER):
    losses1 = []
    ws =[]
    for i in range(size):
        model1.set_params(wr)
        X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
        Y_batch = MakeOneHot(Y_batch, D_out)
        #print('xunl',X_batch.shape)
        # forward, loss, backward, step
        Y_pred = model1.forward(X_batch)
        loss, dout = criterion.get(Y_pred, Y_batch)
        model1.backward(dout)
        w = model1.get_params()
        #ws=uniondict(ws, w)                   #sgd
        ws=uniondict(ws, normalize(lap(4, w))) # ssgd with noise
    SGD(ws, lr=0.1/size, reg=0).step()
    #SGDMomentum(ws, lr=10/(size*1.0002**j), momentum=0.99, reg=0.00000).step()
    #Adam(ws, lr=0.001/size, beta1=0.9, beta2=0.999, epislon=1e-8).step()
    wr=copy.deepcopy(ws)
    if j % 200 == 0:
        print("iter: %s, loss: %s" % (j, loss))
weights = model1.get_params()
with open("weights.pkl","wb") as f:
    pickle.dump(weights, f)
test_size = 1000
print("test_size = " + str(test_size))
for i in range(10):
    X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
    X_test_min,  Y_test_min  = get_batch(X_test,  Y_test,  test_size)

    # TRAIN SET ACC
    Y_pred_min = model1.forward(X_train_min)
    result = np.argmax(Y_pred_min, axis=1) - Y_train_min
    result = list(result)
    print("TRAIN1--> Correct: " + str(result.count(0)) + " out of " + str(X_train_min.shape[0]) + ", acc=" + str(result.count(0)/X_train_min.shape[0]))
    train_avg_acc.append(result.count(0)/X_train_min.shape[0])
    # TEST SET ACC
    Y_pred_min = model1.forward(X_test_min)
    result = np.argmax(Y_pred_min, axis=1) - Y_test_min
    result = list(result)
    print("TEST1--> Correct: " + str(result.count(0)) + " out of " + str(X_test_min.shape[0]) + ", acc=" + str(result.count(0)/X_test_min.shape[0]))
    test_avg_acc.append(result.count(0)/X_test_min.shape[0])
print('ssgd with DP:'+'平均训练精度为',np.average(train_avg_acc))
print('ssgd with DP:'+'平均测试精度为',np.average(test_avg_acc))





