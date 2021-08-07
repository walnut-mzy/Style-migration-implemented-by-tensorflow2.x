## 写在前面的话：

这个模型借鉴了github上一个大佬的模型和他的文章：

[TensorFlow练手项目三：使用VGG19迁移学习实现图像风格迁移_](https://blog.csdn.net/aaronjny/article/details/79681080)

## 食用方法：

在setting.py里面更改配置就可以了

```python
# 内容特征层及loss加权系数
CONTENT_LAYERS = {'block4_conv2': 0.5, 'block5_conv2': 0.5}
# 风格特征层及loss加权系数ge
STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 'block3_conv1': 0.2, 'block4_conv1': 0.2,
                'block5_conv1': 0.2}

#定义所以能用到的层
layer=["block4_conv2","block5_conv2","block1_conv1","block2_conv1","block3_conv1","block4_conv1","block5_conv1"]
# 内容图片路径
CONTENT_IMAGE_PATH = './images_content/content2.jpg'
# 风格图片路径
STYLE_IMAGE_PATH = './images_style/stytle1.jpg'

# 生成图片的保存目录
OUTPUT_DIR = './output1'

# 内容loss总加权系数
CONTENT_LOSS_FACTOR = 1

# 风格loss总加权系数
STYLE_LOSS_FACTOR = 100

# 图片宽度
WIDTH = 527
# 图片高度
HEIGHT = 724

# 训练epoch数
EPOCHS = 20
# 每个epoch训练多少次
STEPS_PER_EPOCH = 100
# 学习率
LEARNING_RATE = 0.03
```

## 训练效果：

训练后的图片：

![在这里插入图片描述](https://img-blog.csdnimg.cn/c6c613b28e0c41088a9722d735e65553.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


**style.jpg**

![在这里插入图片描述](https://img-blog.csdnimg.cn/9489084009dd4244bb07172d931f28e5.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


**content.jpg**

![在这里插入图片描述](https://img-blog.csdnimg.cn/c1ab9eae1d134ed19e888be0ddb08449.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


这个训练结果并不是很满意，但是，通过分析数据集，发现风格图片抽象一点，而content.jpg图片分辨率高一点效果会很好，就比如大佬的图片：


![内容图片2](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0Fhcm9uSm55L25lcnVhbF9zdHlsZV9jaGFuZ2UvbWFzdGVyL3NhbXBsZS9pbnB1dF9jb250ZW50XzIuanBn?x-oss-process=image/format,png)

![img](https://img-blog.csdnimg.cn/img_convert/0fdccba81a24f078670927eb6b6757bb.png)

![生成图片2](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0Fhcm9uSm55L25lcnVhbF9zdHlsZV9jaGFuZ2UvbWFzdGVyL3NhbXBsZS9vdXRwdXRfMi5qcGc?x-oss-process=image/format,png)

这样看起来效果就很不错。