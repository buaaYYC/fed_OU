# Leveraging Critical Learning Period and CLIP-Supervised Training for Enhanced Performance in Heterogeneous Federated Learning
## 目标
- 利用CLIP实现蒸馏学习，知道分类
-  利用CLP，关键学习期，降低通信量
- 

# 训练方式 

## CLIP模型路径
    CLIP 模型会被下载并缓存到你系统的缓存目录中，通常是以下位置：
```
    Linux/MacOS: ~/.cache/clip/
    Windows: C:\Users\<username>\.cache\clip\
```
## 在多个数据集和模型架构上训练
- fedclpCLIP_main.py 是训练主函数
- fedclpCLIP_resnet_main.py 是用来训练并测试resnet 和 cifar100 。
- fedclpCLIP_alexnet_fmnist_main.py 用来训练alexnet和fmnist的，
## 函数说明
- sims.py 中存放的模拟客户端和服务端的函数
- sims_fmnist.py 是针对fmnist，由于类别标签不一致，所以需要正对fmnist类别，CLIP监督学习。
- 其他类似 


## 选择性上传
- 选择性上传：可以选择性的上传数据集，只上传需要的部分数据集，减少通信量。
- 选择性上传的代码实现：
  在 .gitignore 文件中，你可以指定不想提交到远程仓库的文件或目录。示例如下：
  

```
# 忽略单个文件
example.txt

# 忽略某个目录及其所有内容
logs/

# 忽略所有 `.log` 文件
*.log

# 忽略某个特定类型的文件，比如所有 .tmp 文件
*.tmp
```
git add .gitignore

git commit -m "Added .gitignore to exclude specific files"

