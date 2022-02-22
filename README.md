# 1. Chat_robot.py

对话机器人的主要逻辑程序



# 2. NLU/Linear_CLF/train

load_data(data_path):

​	用于加载在data_path位置的数据，仅支持`X y`格式

​	把数据随机打乱

返回值: X,y列表

run(data_path, model_save_path):

​	通过load_data加载数据并训练模型

​	模型为LR+GBDT融合模型

​	将模型分别保存为pkl格式

​	Embedding方法用了比较简单快速的TF-IDF算法,也保存成了pkl格式

# 3. NLU/model_code/classifier_model.py

Classi_Model class:

predict(self, text)方法

返回值:意图