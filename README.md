# FedCEA+CLIP: Federated Learning with Contrastive Embeddings and Language Pretraining
## 目标
- 利用CLIP实现蒸馏学习，知道分类
-  利用CLP，关键学习期，降低通信量
-  

## CLIP模型路径
    CLIP 模型会被下载并缓存到你系统的缓存目录中，通常是以下位置：
```
    Linux/MacOS: ~/.cache/clip/
    Windows: C:\Users\<username>\.cache\clip\
```

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

