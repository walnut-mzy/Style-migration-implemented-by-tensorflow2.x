import setting
import tensorflow as tf
from model import NeuralStyleTransferModel
import setting
import matplotlib.pyplot as plt
import numpy as np
# 我们准备使用经典网络在imagenet数据集上的与训练权重，所以归一化时也要使用imagenet的平均值和标准差
image_mean = tf.constant([0.485, 0.456, 0.406])
image_std = tf.constant([0.299, 0.224, 0.225])
M = setting.WIDTH * setting.HEIGHT
N = 3
model = NeuralStyleTransferModel()
def load_images(image_path, width=setting.WIDTH, height=setting.HEIGHT):
    # 加载文件
    x = tf.io.read_file(image_path)
    # 解码图片
    x = tf.image.decode_jpeg(x, channels=3)
    # 修改图片大小
    x = tf.image.resize(x, [height, width])
    x = x / 255.
    # 归一化()
    x =  (x - image_mean) / image_std
    x = tf.reshape(x, [1, height, width, 3])
    # 返回结果
    return x

def save_image(image, filename):
    #x=image
    x = tf.reshape(image, image.shape[1:])
    x = x * image_std + image_mean
    x = x * 255.
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    x = tf.image.encode_jpeg(x)
    tf.io.write_file(filename, x)
# 加载内容图片
content_image = load_images(setting.CONTENT_IMAGE_PATH)
# 风格图片
style_image = load_images(setting.STYLE_IMAGE_PATH)

# 计算出目标内容图片的内容特征备用
target_content_features = model([content_image, ])['content']
# 计算目标风格图片的风格特征
target_style_features = model([style_image, ])['style']
def compute_content_loss(noise_content_features):
    """
    计算并当前图片的内容loss
    :param noise_content_features: 噪声图片的内容特征
    """

    # 初始化内容损失
    content_losses = []
    # 加权计算内容损失
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        content_loss = tf.reduce_sum(tf.square(noise_feature - target_feature))
        # 计算系数
        x = 2. * M * N
        layer_content_loss = content_loss / x
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)

def gram_matrix(feature):
    """
    计算给定特征的格拉姆矩阵
    """
    # 先交换维度，把channel维度提到最前面
    x = tf.transpose(feature, perm=[2, 0, 1])
    # reshape，压缩成2d
    x = tf.reshape(x, (x.shape[0], -1))
    # 计算x和x的逆的乘积
    return x @ tf.transpose(x)
def compute_style_loss(noise_style_features):
    """
    计算并返回图片的风格loss
    :param noise_style_features: 噪声图片的风格特征
    """
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
        noise_gram_matrix = gram_matrix(noise_feature)
        style_gram_matrix = gram_matrix(target_feature)
        style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
        # 计算系数
        x = 4. * (M ** 2) * (N ** 2)
        layer_style_loss =style_loss / x
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)
def total_loss(noise_features):
    """
    计算总损失
    :param noise_features: 噪声图片特征数据
    """
    content_loss = compute_content_loss(noise_features['content'])
    style_loss = compute_style_loss(noise_features['style'])
    return content_loss * setting.CONTENT_LOSS_FACTOR + style_loss * setting.STYLE_LOSS_FACTOR