# coding:utf-8
from ultralytics import YOLOv10

# 模型配置文件
model_yaml_path = "/var/doc/yolov10/ultralytics/cfg/models/v10/yolov10n.yaml"
# 预训练模型
pre_model_name = '/var/doc/yolov10/mydata/yolov10n.pt'

# 数据集配置文件
data_yaml_path = '/var/doc/yolov10/mydata/helmet.yaml'

if __name__ == '__main__':
	# 加载预训练模型
	model = YOLOv10(model_yaml_path).load(pre_model_name)
	# 训练模型
	results = model.train(data = data_yaml_path,
		epochs = 1,                                  #训练次数推荐 100
		batch = 8,
		name = '/var/doc/yolov10/mydata/train_v10'   # 保存路径
	)