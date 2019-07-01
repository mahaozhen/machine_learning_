import numpy as np
from Cluster import Cluster
#定义一个足够抽象的基类以囊括所有我们关心的算法
class CvDNode:
    """
    初始化结构
    self._x, self._y : 记录数据集的变量
    self.base, self.chaos : 记录对数的底和当前的不确定性
    self.criterion, self.category : 记录该node计算信息增益的方法和所属的类别
    self.left_child, slef.right_child : 针对连续型特征和CART、记录该node的左右子节点
    self.chrildren, self.leafs : 记录该node的所有子节点和所有下属的叶节点
    self.sample_weight : 记录样本权重
    self.wc : 记录各个维度的特征是否连续的列表（whether continuous的缩写）
    self.tree : 记录该node所属的tree
    self.feature_dim, self._tar, self.feats : 记录该node划分标准的相关信息，
    分别是记录作为划分标准的特征所对应的的维度j*；针对连续型特征和CART，记录二分标准；
    记录该node进行选择的、作为划分标准的特征的维度
    self.parent, self.is_root : 记录该node的父节点以及该节点是否为根节点
    self._depth,self.prev_feat : 记录node的深度和其父节点的划分标准
    self.is_cart : 记录该node是否使用了CART算法
    self.is_contionous : 记录该node选择的划分标准对应的特征是否连续
    self.pruned : 记录该node是否已被剪掉，后面是西安局部剪枝算法时会用到
    """
    def __init__(self, tree=None, base=2, chaos=None, depth=0, parent=None, is_root=True, prev_feat="Root"):
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}
        self.sample_weight = None
        self.wc = None
        self.tree = tree
        #如果传入了tree的话就进行相应的初始化
        if tree is not None:
            #由于数据处理是有tree完成的
            #所以各个维度的特征是否是连续型随机变量也是tree记录的
            self.wc = tree.whether_continuous
            #这里的nodes变量是tree中记录所有node的列表
            tree.nodes.appends(self)
        self.feature_dim, self.tar, self.feats = None, None, []
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.pruned = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    #重载__lt__方法，使得node之间可以比较谁更小，进而方便调试和可视化
    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    #重载__str__ 和 __repr__方法，同样是为了方便调试和可视化
    def __str__(self):
        if self.category is None:
            return "CvDNode ({})  ({} -> {})".format(self._depth,self.prev_feat,self.feature_dim)
        return "CvDNode ({})  ({} -> class: {})".format(self._depth, self.prev_feat, self.tree.label_dic[self.category])
    __repr__ = __str__

    #定义children属性，主要是区分开连续+CART的情况和其余情况
    #有了该属性后，想要获得所有子节点时就不用分情况讨论了
    @property
    def children(self):
        return {"left": self.left_child, "right": self.right_child} if (self.is_cart or self.is_continuous) else self._children

    #递归定义height属性
    #叶节点高度定义为1，其余节点的高度定义为最高的字节点的高度+1
    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height if _child is not None else 0 for _child in self.children.values()])

    #定义info_dic（信息字典）属性，它记录了该node的主要信息
    #在更新各个node的叶节点时，被记录进各个self.leafs属性的就是该字典
    @property
    def info_dic(self):
        return {"chaos": self.chaos, "y": self._y}

    #定义第一种停止准则：当特征维度为0或当前node的数据的不确定性小于阈值e时停止
    #同时，如果用户制定了决策树的最大深度，那么当该node的深度太深时也停止
    #若满足了停止条件，该函数会返回True，否则会返回false
    def stop1(self, eps):
        if(
            self._x.shape[1] == 0 or (self.chaos is not None and self.chaos <= eps) or (self.tree.max_depth is not None and self._depth >= self.tree.max_depth)
        ):
            #调用处理停止情况的方法
            self._handle_terminate()
            return True
        return False

    #定义第二种停止准则，当最大信息增益仍然小于阈值e时停止
    def stop2(self, max_gain, eps):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False

    #利用bincount方法定义根据数据生成node所属类别的方法
    def get_category(self):
        return np.argmax(np.bincount(self._y))

    #定义处理停止情况的方法，核心思想就是把该node转化为一个叶节点
    def _handle_terminate(self):
        #首先要生成该node所属的类别
        self.category = self.get_category()
        #然后一路回溯，更新父节点，父节点的父节点，等到，记录叶节点的属性leafs
        _parent = self.parent
        while _parent is not None:
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent

    def prune(self):
        #调用相应方法进行计算该node所属类别
        self.category = self.get_category()
        #记录由于该node转化为叶节点而被剪去的、下属的叶节点
        _pop_lst = [key for key in self.leafs]
        #然后一路回溯，更新各个parent的属性leafs（使用id作为key以避免重复）
        _parent = self.parent
        while _parent is not None:
            for _k in _pop_lst:
                #删去由于局部剪枝而被剪掉的叶节点
                _parent.leafs.pop(_k)
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent
        #调用mark_pruned方法将自己所有的子节点、子节点的子节点、、、的pruned属性设置为true，因为他们都被剪掉了
        self.mark_pruned()
        #重置各个属性
        self.feature_dim = None
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}

    def mark_pruned(self):
        self.pruned = True
        #遍历各个子节点


