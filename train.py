import tensorflow as tf
from tensorflow import optimizers
import setting
import numpy as np
import os
import utils1
from tqdm import tqdm
# 使用Adam优化器
optimizer = tf.keras.optimizers.Adam(setting.LEARNING_RATE)
model=utils1.model
# 基于内容图片随机生成一张噪声图片
#noise_image = tf.Variable((utils1.content_image + np.random.uniform(-0.2, 0.2, (1, setting.HEIGHT, setting.WIDTH, 3))) / 2)
noise_image =tf.Variable(utils1.content_image)

# 使用tf.function加速训练
@tf.function
def train_one_step():
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)

        loss = utils1.total_loss(noise_outputs)
    # 求梯度
    grad = tape.gradient(loss, noise_image)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients([(grad, noise_image)])
    return loss

if __name__ == '__main__':

    # 创建保存生成图片的文件夹
    if not os.path.exists(setting.OUTPUT_DIR):
        os.mkdir(setting.OUTPUT_DIR)

    # 共训练settings.EPOCHS个epochs
    for epoch in range(setting.EPOCHS):
        # 使用tqdm提示训练进度
        with tqdm(total=setting.STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(epoch, setting.EPOCHS)) as pbar:
            # 每个epoch训练settings.STEPS_PER_EPOCH次
            for step in range(setting.STEPS_PER_EPOCH):
                _loss = train_one_step()
                pbar.set_postfix({'loss': '%.4f' % float(_loss)})
                pbar.update(1)
            # 每个epoch保存一次图片
            utils1.save_image(noise_image, '{}/{}.jpg'.format(setting.OUTPUT_DIR, epoch + 1))
