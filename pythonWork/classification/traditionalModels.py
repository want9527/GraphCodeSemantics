import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error,f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

from imblearn.combine import SMOTETomek

#IMBALANCE_PROCESSOR = RandomOverSampler(random_state=42)
#IMBALANCE_PROCESSOR = BalancedRandomForestClassifier(random_state=42)  # RandomOverSampler(), RandomUnderSampler(), None, 'cost' 创新点（处理类不平衡问题）用embedding集成学习方法
#IMBALANCE_PROCESSOR = BalancedBaggingClassifier(random_state=42)  # RandomOverSampler(), RandomUnderSampler(), None, 'cost' 创新点（处理类不平衡问题）用embedding集成学习方法
#IMBALANCE_PROCESSOR = SMOTEENN(random_state=42)
IMBALANCE_PROCESSOR = SMOTETomek()

def train(outPath,pathArray,count=None):
    #for path in pathArray:
        # 读取CSV文件
        data = pd.read_csv(pathArray)

        # 特征和标签
        X = data.iloc[:, :-1].values  # 所有行，除了最后一列的所有列
        y = data.iloc[:, -1].values  # 所有行的最后一列

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X, y = IMBALANCE_PROCESSOR.fit_resample(X, y)
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []

        models = [
            LogisticRegression(max_iter=200),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=100),
            SVC(kernel='linear'),
            GaussianNB(),
            KNeighborsClassifier(n_neighbors=5),
            Perceptron(max_iter=10, tol=None),
            GradientBoostingClassifier(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            MLPClassifier(max_iter=1000)
        ]

        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            loss = mean_squared_error(y_test, y_pred)
            f1 = f1_score(y_test , y_pred)
            a = roc_auc_score(y_test, y_pred)

            print(f"Model: {model.__class__.__name__}")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Loss: {loss:.2f}\n")
            print(f"f1: {f1:.2f}\n")
            print(f"auc: {a:.2f}\n")

            results.append({
                "Model": model.__class__.__name__,
                "Accuracy": accuracy,
                "Loss": loss,
                "F1": f1,
                "AUC": a
            })

        # 将结果转换为DataFrame
        df_results = pd.DataFrame(results)

        # 保存到Excel文件
        pathSplit = pathArray.split("/")
        # 用count判断是否批量执行
        if(count == None):
            output_path = outPath + pathSplit[len(pathSplit) - 1] + ".xlsx"
        else:
            output_path = outPath + pathSplit[len(pathSplit) - 1] + str(count) + ".xlsx"
        output_path = output_path.replace(".csv", "")
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, index=False, sheet_name='Results')

        print(f"Results saved to {output_path}")


firstOutPath = "CFGResult/"
afterOutPath = "CFGResult/after/"
firstPath = "../out/featureVec/CFG/xalan-j_2_6_0-first2one.csv"

# 批量执行次数
count = 1

# for i in range(count):
#     firstPath = "../out/featureVec/CFG/xalan-2.6.csv"
#     train(firstOutPath, firstPath, 100 + i)

train(firstOutPath, firstPath)