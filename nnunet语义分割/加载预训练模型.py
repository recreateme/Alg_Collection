import pickle

import torch


model = torch.load(r"D:\Work\data\RAOS-Real\trained_weights\nnUNetV1\all\model_final_checkpoint.model", map_location=torch.device('cpu'))
# print(model)

# pickle模型配置的保存路径
path = r"D:\Work\data\RAOS-Real\trained_weights\nnUNetV1\all\plans.pkl"
with open(path, 'rb') as f:
    plans = pickle.load(f)
keys = plans.keys()
# print(plans[keys[0]])
print(keys)