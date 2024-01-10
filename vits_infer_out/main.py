"""
XGBoost被广泛的应用于工业界，LightGBM有效的提升了GBDT的计算效率;
而 CatBoost 号称是比XGBoost和LightGBM在算法准确率等方面表现更为优秀的算法
"""
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV


"""
对于gbdt的调参，一点建议，tree的数量通过earlystopping的功能来决定即可，对于整个gbdt模型的影响最大的参数，
一个是tree的数量，一个是max_depth深度，一个是行列采样的比例
"""
cat = CatBoostClassifier(
    iterations=5000,     # 可以建立的最大树数。当使用其他限制迭代次数的参数时，树的最终数量可能少于此参数中指定的数量。
    verbose=0,
    random_seed=6543,
    l2_leaf_reg=4.6591278779517808,      # 6.6591278779517808, default:3
    learning_rate=0.005599066836106983,  # 默认情况下，学习率是根据数据集属性和迭代次数自动定义的
    subsample = 0.35,
    allow_const_label=True,
    loss_function = 'CrossEntropy', 
    eval_metric = 'AUC',
    early_stopping_rounds=3000,
)