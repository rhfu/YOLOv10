# coding:utf-8
from ultralytics import YOLOv10

# 根据自己项目路径需要更换
path = "/var/doc/yolov10/"

# 模型配置文件
model_yaml_path = path + "ultralytics/cfg/models/v10/yolov10n.yaml"

# 预训练模型
pre_model_name = path + 'mydata/yolov10n.pt'

# 数据集配置文件
data_yaml_path = path + 'mydata/helmet.yaml'

if __name__ == '__main__':
    
	# 加载预训练模型
	model = YOLOv10(model_yaml_path).load(pre_model_name)
 
	# 训练模型
	results = model.train(data = data_yaml_path,
		epochs = 1,                        #训练次数推荐 100
		batch = 8,
		name = path + 'mydata/train_v10'   # 保存路径
	)
