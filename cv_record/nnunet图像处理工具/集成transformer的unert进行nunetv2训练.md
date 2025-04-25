# 结合nnunetv2的自适应训练swin-unetr分割

## 1 源码安装（不能直接pip）

```
# 创建虚拟环境
conda create local_seg python=3.10
activate local_seg

#  通过获取nnUNetv2源码，执行完成依赖安装(内置pytorch等依赖)
git clone https://github.com/MIC-DKFZ/nnUNet.git
pip install -e .
# 安装额外的组件
pip install monai==1.2.0 hiddenlayer==0.3.0
```

## 2 数据集结构规范（预处理工具规范）

**（这一步结合pyqt的图像处理工具得到）**

1.  在`nnUNet_raw`目录下创建`DatasetXXX_CTSeg`（XXX为三位数ID），包含：

  ◦ `imagesTr/`: 训练CT图像（命名格式：case_0000.nii.gz）

  ◦ `labelsTr/`: 对应分割标签（命名格式：case.nii.gz）

  ◦ `dataset.json`: 定义通道名称、标签类别等元信息

```json
  {
      "channel_names": {"0": "CT"},
      "labels": {"background":0, "tumor":1},
      "numTraining": 100,
      "file_ending": ".nii.gz"
  }
```

2. 环境变量配置

   临时环境变量设置：

   ```
   # Linux
   export nnUNet_raw="/path/to/raw_data"
   export nnUNet_preprocessed="/path/to/preprocessed"
   export nnUNet_results="/path/to/results"
   
   # windows cmd
   set nnUNet_raw=C:\path\to\raw
   set nnUNet_preprocessed=C:\path\to\preprocessed
   set nnUNet_results=C:\path\to\results
   ```

   

## 3 模型集成与训练

### 3.1自定义Swin-UNETR架构适配器

• 在`nnUNet/training/nnUNetTrainer`目录下新建`nnUNetTrainerSwinUNETR.py`

```python
  from monai.networks.nets import SwinUNETR
  
  class nnUNetTrainerSwinUNETR(nnUNetTrainer):
      def __init__(self, plans, configuration, fold, dataset_json):
          super().__init__(plans, configuration, fold, dataset_json)
          self.network = SwinUNETR(
              img_size=(96,96,96), 
              in_channels=1, 
              out_channels=2,
              feature_size=48,
              use_checkpoint=True
          )
```

### 3.2 训练参数配置

• 启动数据预处理：`nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity`

• 关键训练命令：

```bash
  nnUNetv2_train XXX 3d_fullres 0 \
      -tr nnUNetTrainerSwinUNETR \
      --npz --batch_size 2 \
      --lr 1e-4 --epochs 500 \
      -device cuda --amp
```
• 参数说明：

  ◦ `-tr`指定自定义训练器

  ◦ `--amp`启用混合精度训练加速

  ◦ `--npz`保存softmax输出用于模型集成

### 3.3 多GPU训练优化

• 使用`CUDA_VISIBLE_DEVICES=0,1`指定可见GPU

• 分布式训练启动命令：

```bash
  torchrun --nproc_per_node=2 nnUNetv2_train ... \ --distributed=ddp
```

---

## 4 推理验证与性能优化

1. 模型验证策略（默认配置）
   • 五折交叉验证：通过`-f all`参数运行全部折数

   • 最优模型选择：`nnUNetv2_find_best_configuration XXX -c 3d_fullres`


2. 部署方案
   • 导出ONNX格式：

     ```python
     torch.onnx.export(model, dummy_input, "swinunetr.onnx",opset_version=17, 
                       dynamic_axes={'input':{0:'batch'},'output':{0:'batch'}})
     ```
   • TensorRT加速：
   
  ```bash
     trtexec --onnx=swinunetr.onnx \
         --saveEngine=swinunetr.plan \
         --fp16 --workspace=4096
     ```
   


---

## 5 典型问题

1. 显存不足问题
   • 降低`batch_size`至1-2

   • 启用梯度累积：`--gradient_accumulation_steps 4`（梯度累积在大批量的前提下增大了batch_size并减小了显存需求，用时间换空间）


2. 标签不匹配错误
   • 检查`dataset.json`中`labels`字段是否覆盖所有标注类别，不能多也不能少（运行`nnUNetv2_verify_dataset_integrity -d XXX`验证数据一致性）

