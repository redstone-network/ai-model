import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from enum import Enum, unique
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from typing import List, Dict, Union, Any


@unique
class ModelType(Enum):
    # 非监督学习模型
    IFOREST = 0

    # 监督学习模型
    LGBM = 10

def preprocessing(origin_data: Union[List[Dict[str, Any]], str]):
    """
    数据预处理模块
    :param origin_data: 输入的原始数据，格式如下，支持python标准数据结构、json字符串、json文件路径
    [
      {
        "blockhigh": 1487575,
        "hash": "0x0d65599bf9d2b158d3c5aaef7af220c908ee46d862a077f29a0efc3dcc191f0d",
        "timestamp": 1584438762,
        "module": "balances",
        "call": "transfer",
        "from": "Dkukcu2gosh9n4C9BuaJwm6nfsweGcN1uJ2tJWyUwyJ6hrY",
        "to": "JFArxqV6rqPSwBok3zQDnj5jL6vwsZQDwYXXqb1cFygnYVt",
        "balance": "900000000000",
        "flag": false
      },
      ...
    ]
    :return:
        :X 是预处理完后的特征数据，类型为numpy array
        :y 是打标的标签数据，如果原始数据中未包含flag字段，则输出为None
    """
    if type(origin_data) == str:
        data = pd.read_json(origin_data)
    elif type(origin_data) == list:
        data = pd.DataFrame(origin_data)
    else:
        raise TypeError("input data type error")

    X, y = None, None
    if 'flag' in data.columns:
        data.flag = data.flag.astype(int)
        y = data.flag.values
    data.balance = data.balance / 10 ** 10
    data['hour'] = data.timestamp.dt.hour

    X = data[['balance', 'blockhigh', 'hour']].values
    return X, y


def train(X: np.ndarray, y: np.ndarray, model_type: ModelType = ModelType.IFOREST,
          saved_path: str = './model.txt'):
    """
    训练模型模块
    :param X: 特征数据
    :param y: 标签数据
    :param model_type: 模型类型
    :param saved_path: 模型训练完后保存的位置
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=666)

    if model_type == ModelType.IFOREST:
        model = IsolationForest(n_estimators=200, max_features=1.0, random_state=666).fit(X)
        ypred = model.predict(X_test)
        ypred[ypred == -1] = 0
        print(f"accuracy score: {accuracy_score(y_test, ypred)}")
        print(f"auc score: {roc_auc_score(y_test, ypred)}")
        joblib.dump(model, saved_path)

    elif model_type == ModelType.LGBM:
        param = {'num_leaves': 31, 'objective': 'binary', 'metric': ['auc', 'binary_logloss'], 'learning_rate': 0.01}
        train_set = lgb.Dataset(X_train, label=y_train)
        valid_sets = [lgb.Dataset(X_test, label=y_test)]
        model = lgb.train(param, train_set, num_boost_round=10, valid_sets=valid_sets, early_stopping_rounds=5)
        model.save_model(saved_path, num_iteration=model.best_iteration)

    else:
        raise ValueError("invalid model")


def load(model_type: ModelType = ModelType.IFOREST, saved_path: str = './model.txt'):
    """
    模型加载模块
    :param model_type: 模型类型
    :param saved_path: 模型文件的位置
    :return model: 加载后的模型对象
    """
    if model_type == ModelType.IFOREST:
        model = joblib.load(saved_path)
    elif model_type == ModelType.LGBM:
        model = lgb.Booster(model_file=saved_path)
    else:
        raise ValueError("invalid model")
    return model


def inference(X: np.ndarray, model: Union[lgb.Booster, IsolationForest]):
    """
    模型推理模块
    :param X: 特征数据
    :param model: 模型对象
    :return ypred: 预测的结果
    """
    if type(model) == IsolationForest:
        ypred = model.predict(X)
        ypred[ypred == -1] = 0
    elif type(model) == lgb.Booster:
        ypred = model.predict(X)
    else:
        raise ValueError("invalid model")
    return ypred


if __name__ == "__main__":
    normal = pd.read_json("./normal/normal-output.json")
    multi = pd.read_json("./multi/multi-product.json")
    big = pd.read_json("./big/big-product.json")
    total = pd.concat([normal, multi, big])
    total.reset_index(drop=True, inplace=True)
    total.to_json("./total.json", orient="records")

    X_data, y_data = preprocessing("./total.json")
    # train(X_data, y_data, model_type=ModelType.IFOREST, saved_path="./iforest.model")
    # iforest = load(model_type=ModelType.IFOREST, saved_path="./iforest.model")
    # result1 = inference(X_data, iforest)
    # result1[result1 < 0.5] = 0
    # result1[result1 >= 0.5] = 1
    # print(f"accuracy score: {accuracy_score(y_data, result1)}")
    # print(f"auc score: {roc_auc_score(y_data, result1)}")
    # print(f"f1 score: {f1_score(y_data, result1)}")
    # print(f"recall score: {recall_score(y_data, result1)}")

    train(X_data, y_data, model_type=ModelType.LGBM, saved_path="./lgbm.model")
    lgbm = load(model_type=ModelType.LGBM, saved_path="./lgbm.model")
    result2 = inference(X_data, lgbm)
    result2[result2 < 0.065] = 0 #
    result2[result2 >= 0.065] = 1 #预测为负样本的概率，阈值可以通过 负样本 / 正样本 来设置
    print(f"accuracy score: {accuracy_score(y_data, result2)}")
    print(f"auc score: {roc_auc_score(y_data, result2)}")
    print(f"f1 score: {f1_score(y_data, result2)}")
    print(f"recall score: {recall_score(y_data, result2)}")
