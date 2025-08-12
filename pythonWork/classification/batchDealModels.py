from classification.traditionalModels import train
from classification.utils import firstPathCFGMap,afterPathCFGMap,firstPathPDGMap,afterPathPDGMap
import pandas as pd
import os
import numpy as np

firstPathMap = firstPathCFGMap
# afterPathMap = afterPathCFGMap
firstOutRootPath = "./CFGResult/first/"
# afterOutRootPath = "./CFGResult/after/"

# firstPathMap = firstPathPDGMap
# afterPathMap = afterPathPDGMap
# firstOutRootPath = "./PDGResult/first/"
# afterOutRootPath = "./PDGResult/after/"


# 批量执行次数
count = 1

for i in range(count):
    for key, value in firstPathMap.items():
        firstOutPath = firstOutRootPath + key + str(i) + "/"
        # 检查文件夹是否存在
        if not os.path.exists(firstOutPath):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(firstOutPath)
            print(f"文件夹 {firstOutPath} 已创建。")
        else:
            print(f"文件夹 {firstOutPath} 已存在。")
        train(firstOutPath, value, i)
    # for key, value in afterPathMap.items():
    #     afterOutPath = afterOutRootPath + key + str(i) + "/"
    #     # 检查文件夹是否存在
    #     if not os.path.exists(afterOutPath):
    #         # 如果文件夹不存在，则创建文件夹
    #         os.makedirs(afterOutPath)
    #         print(f"文件夹 {afterOutPath} 已创建。")
    #     else:
    #         print(f"文件夹 {afterOutPath} 已存在。")
    #     train(afterOutPath, value, i)

def dealAvg(path_array, output_file):

    # 初始化一个空的列表，用于存储所有文件的DataFrame
    df_list = []
    header_row = None
    header_col = None

    # 循环获取指定目录下的所有Excel文件
    for path in path_array:
        file = os.listdir(path)
        try:
            # 拼接文件路径
            file_path = os.path.join(path, file[0])

            # 读取整个Excel文件以保留表头信息
            df_full = pd.read_excel(file_path, header=0)
            if header_row is None:
                header_row = df_full.columns.tolist()  # 保留第一行表头信息
            if header_col is None:
                header_col = df_full.iloc[:, 0].tolist()  # 保留第一列表头信息

            # 读取指定区域（第二行第二列到第12行第5列）
            df = pd.read_excel(file_path, usecols="B:E",  nrows=11)
            data_array = df.values.tolist()
            # 将DataFrame添加到列表中
            df_list.append(data_array)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    if df_list:
        average_array = np.mean(df_list, axis=0)
        # 创建一个新的DataFrame，首先设置header_row作为列名
        df = pd.DataFrame(columns=header_row)

        # 设置header_col作为第一列
        df[header_row[0]] = header_col

        # 将average_array插入到DataFrame中，从第二行第二列开始
        for i in range(len(average_array)):
            for j in range(len(average_array[i])):
                df.iat[i, j + 1] = average_array[i, j]

        # 将DataFrame写入到新的Excel文件中
        df.to_excel(output_file, index=False)
        print(f"处理完成，结果已存储在'{output_file}'中")
    else:
        print("未找到有效的Excel文件")


path_first = "./CFGResult/first/"
# path_after = "./CFGResult/after/"
#
# path_first = "./PDGResult/first/"
# path_after = "./PDGResult/after/"


for key, value in firstPathMap.items():
    path_first_array = []
    path_after_array = []
    for i in range(count):
        path_first_array.append(path_first + key + str(i) + "/")
        dealAvg(path_first_array, path_first + key + str(i) + "average_result_first.xlsx")

# for key, value in afterPathMap.items():
#     path_first_array = []
#     path_after_array = []
#     for i in range(count):
#         path_after_array.append(path_after + key + str(i) + "/")
#         dealAvg(path_after_array, path_after + key + str(i) + "average_result_after.xlsx")






