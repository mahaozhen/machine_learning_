import numpy as np
#定义朴素贝叶斯模型的基类，方便以后的拓展
class NaiveBayes:
    """
    初始化结构
    self._x, self,_y ：记录训练集的变量
    self._data ：核心数组，存储实际使用的条件概率的相关信息
    self._func ：模型核心--决策函数，能够根据输入的x、y输出对应的后验概率
    self._n_possibilities ：记录各个维度特征取值个数的数组：[S1,S2,...,Sn]
    self._labelled_x ：记录按类别分开后的输入数据的数组
    self._label_zip ：记录类别相关信息的数组，视具体算法，定义会有所不同
    self._cat_counter ：核心数组，记录第i类数据的个数（cat是category的缩写）
    self._con_counter ：核心数组，记录数据条件概率的原始极大似然估计
        self._con_counter[d][c][p] = p(X(d)=p|y=c) （con是conditional的缩写）
    self._label_dic ：核心字典，用于记录数值化类别时的转换关系
    self._feat_dics ：核心字典，用于记录数值化各维度特征时的转换关系
    """
    def __init__(self):
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dic = self._feat_dics = None

    #重载__getitem__运算符以避免定义大量property
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self,"_" + item)

    #留下抽象方法让子类定义，这里的tar_idx参数和self._tar_idx的意义一致
    def feed_data(self, x, y, sample_weight=None):
        pass

    #留下抽象方法让子类定义，这里的sample_weight参数代表着样本权重
    def feed_sample_weight(self, sample_weight=None):
        pass


    #模型训练

    #定义计算先验概率的函数，lb就是各个估计中的平滑项人
    #lb的默认值是1，也就是说默认采取拉普拉斯平滑
    def get_prior_probability(self, lb=1):
        return [(_c_num + lb) / (len(self._y) + lb * len(self._cat_counter))
                for _c_num in self._cat_counter]

    #定义具有普适性的训练函数
    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        # 如果有传入x,y，那么就用传入的x,y初始化模型
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        #调用核心算法得到决策函数
        self._func = self._fit(lb)

    #留下抽象核心算法让子类定义
    def _fit(self, lb):
        pass


    #模型的预测和评估

    #定义预测单一样本的函数
    #参数get_raw_result控制该函数是输出预测的类别还是输出相应的后验概率
    #get_raw_result=False则输出类别，get_raw_result=True，则输出后验概率
    def predict_one(self, x, get_raw_resule=False):
        #在进行预测之前，要先把新的输入数据数值化
        #如果输入的是numpy数组，要先将它转换成python数组
        #因为python数组在数值化操作上更快
        if isinstance(x, np.ndarray):
            x = x.tolist()
        #否则，对数组进行拷贝
        else:
            x = x[:]
        #调用相关方法进行数值化，该方法随具体模型的不同而不同
        x = self._transfer_x(x)
        m_arg, m_probability = 0, 0
        #遍历各类别、找到能使后验概率最大化的类别
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if not get_raw_resule:
            return self.label_dic[m_arg]
        return m_probability

    #定义预测多样本的函数，本质是不断调用上面定义的predict_one函数
    def predict(self, x, get_raw_result=False):
        return np.array(self.predict_one(xx, get_raw_result) for xx in x)

    #定义能对新数据进行评估的方法，这里暂以简单的输出准确率作为演示
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        print("Acc: {:12.6} %".format(100 * np.sum(y_pred == y) / len(y)))










