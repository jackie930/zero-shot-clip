# zero-shot-clip
Experiment on clip vs classic pretrained image classsification models

## comparison task, compare clip vs. traditional method (build on torchvision)

```shell script
source activate pytorch_p36
#prepare data
python source/preprocess.py

# the script will run training over "resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception" for 50 epcoh each, and print out accuracy on same dataset
python source/train.py
```

result：


"resnet", 
Train Loss: 0.0871 Acc: 0.9681
Train class 正常图片 accuracy 1.0
Train class 缺失图片 accuracy 0.8958333333333334
Validation Loss: 0.0201 Acc: 1.0000
Validation class 正常图片 accuracy 1.0
Validation class 缺失图片 accuracy 1.0

Training complete in 5m 33s

"alexnet", 
Train Loss: 0.0660 Acc: 0.9787
Train class 正常图片 accuracy 1.0
Train class 缺失图片 accuracy 0.8666666666666668
Validation Loss: 0.0092 Acc: 1.0000
Validation class 正常图片 accuracy 1.0
Validation class 缺失图片 accuracy 1.0

Training complete in 5m 32s

"vgg", 
Train Loss: 0.1672 Acc: 0.9468
Train class 正常图片 accuracy 1.0
Train class 缺失图片 accuracy 0.8055555555555555
Validation Loss: 0.0699 Acc: 0.9565
Validation class 正常图片 accuracy 1.0
Validation class 缺失图片 accuracy 0.8333333333333333

Training complete in 5m 35s

"squeezenet", 
Train Loss: 0.1063 Acc: 0.9681
Train class 正常图片 accuracy 1.0
Train class 缺失图片 accuracy 0.9166666666666667
Validation Loss: 0.0161 Acc: 1.0000
Validation class 正常图片 accuracy 1.0
Validation class 缺失图片 accuracy 1.0

Training complete in 5m 32s

"densenet", 
Train Loss: 0.1045 Acc: 0.9468
Train class 正常图片 accuracy 1.0
Train class 缺失图片 accuracy 0.8416666666666666
Validation Loss: 0.0372 Acc: 1.0000
Validation class 正常图片 accuracy 1.0
Validation class 缺失图片 accuracy 1.0

Training complete in 5m 39s

"inception"
Train Loss: 0.1625 Acc: 0.9681
Train class 正常图片 accuracy 1.0
Train class 缺失图片 accuracy 0.861111111111111
Validation Loss: 0.0554 Acc: 1.0000
Validation class 正常图片 accuracy 1.0
Validation class 缺失图片 accuracy 1.0


