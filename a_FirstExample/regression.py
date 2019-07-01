import numpy as np
import matplotlib.pyplot as plt

#获取与处理数据

#定义存储输入数据（x）和目标数据（y）的数组
x, y = [], []
#遍历数据集，变量sample对应的是一个个样本
for sample in open("./_Data/prices.txt","r"):
    #由于数据是用逗号隔开的，所以调用python中的split方法并将逗号作为参数传入
    _x, _y = sample.split(",")
    #将字符串数据转化为浮点数
    x.append(float(_x))
    y.append(float(_y))

#读取完数据后，将他们转化为numpy数组以方便进一步处理
x, y = np.array(x), np.array(y)
#标准化
x = (x - x.mean()) / x.std()

#将原始数据以散点图的形式画出
plt.figure()
plt.scatter(x, y, c="g", s=6)
plt.show()


#选择与训练模型

#在（-2,4）区间上取100个点作为画图的基础
x0 =  np.linspace(-2, 4, 100)
#利用numpy函数定义训练并返回多项式回归模型的函数
#deg参数代表着模型参数中的n，即模型中多项式的次数
#返回的模型能够根据输入的x（默认是x0）。返回相对应的预测的y
def get_model(deg):
    return lambda input_x = x0: np.polyval(np.polyfit(x, y, deg), input_x)


#评估与可视化结果

#根据参数n、输入的x、y返回相对应的损失
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()
#定义测试参数集并根据它进行各种实验
test_set = (1, 4, 10)
for d in test_set:
    print(get_cost(d, x, y))
    #输出相应的损失


#画出相应的图像
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))
#将横轴、纵轴的范围分别限制在（-2,4）、（10的5次方，8*10的5次方）
plt.xlim(-2, 4)
plt.ylim(1e5,8e5)
#调用legend使曲线对应的label正确显示
plt.legend()
plt.show()

