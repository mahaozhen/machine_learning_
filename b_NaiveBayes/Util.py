import numpy as np
class DataUtil:
    #定义一个方法使其能从文件中读取数据
    #该方法接收五个参数：数据集的名字，数据集的路径，训练样本数，类别所在列，是否打乱数据
    def get_dataset(name, path, train_num=None, tar_idx=None, shuffle=True):
        x = []
        #将编码设为utf8以便读入中文等特殊字符
        with open(path, "r", encoding="utf8") as file:
            #如果是气球数据集的话，直接依逗号分割数据即可
            if "balloon" in name:
                for sample in file:
                    x.append(sample.strip().split(","))
        #默认打乱数据
        if shuffle:
            np.random.shuffle(x)
        #默认类别在最后一列
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx)for xx in x])
        x = np.array(x)
        #默认全都是训练样本
        if train_num is None:
            return x, y
        #若传入了训练样本数，则依之将数据集切分为训练集和测试集
        return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])

