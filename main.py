# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys
from  tensorflow.python import keras
from tensorflow.python.keras.layers import Dense,Flatten,Conv2D,GlobalAveragePooling2D,MaxPooling2D,Concatenate,UpSampling2D
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.activations import softmax

import tensorflow.contrib.slim as slim
from input_utils import input_fn
from tnet import TNet
from mnet import MNet
from hook_utils import CheckpointSaverHook



def main_model_fn(features,labels,mode,params):
    print(features)
    tnet = TNet()
    tnet_preprocessed_dict = tnet.preprocess(features)
    image_concat = tnet_preprocessed_dict['image_concat']
    tnet_predict_output=  tnet.predict(image_concat, mode == tf.estimator.ModeKeys.TRAIN)
    tnet_postprocessed_dict = tnet.postprocess(tnet_preprocessed_dict, tnet_predict_output)

    mnet = MNet()
    mnet_preprocessed_dict = mnet.preprocess(tnet_postprocessed_dict)
    mnet_input = mnet_preprocessed_dict['mnet_input']
    mnet_predict_output=  mnet.predict(mnet_input, mode == tf.estimator.ModeKeys.TRAIN)
    mnet_postprocessed_dict = mnet.postprocess(mnet_preprocessed_dict, mnet_predict_output)

    features_fused = mnet_postprocessed_dict["features_fused"]
    labels = tf.cast(labels, tf.float32)
    loss = tf.reduce_mean(tf.square(features_fused - labels))

    #如果是预测模式，直接返回结果
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"result":tf.argmax(Flatten()(features_fused),1)}
        )

    #优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    #定义训练过程。传入global_step的目的，为了在TensorBoard中显示图像的横坐标
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    tf.summary.scalar('loss', loss)
    #定义评测标准
    #这个函数会在调用Estimator.evaluate的时候调用
    # a = Flatten()(features_fused)
    # b =  Flatten()(labels)

    accuracy = tf.metrics.accuracy(
            predictions=features_fused,
            labels=labels,
            name="acc_op"
    )
    eval_metric_ops = {
        "my_metric":accuracy
    }

    # print("predicted features after TNET: ", tnet_output.get_shape())
    # print("tensor of labels: ",labels.get_shape())

    restoreCheckpointHook = my_restore_hook()
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=tf.losses.get_total_loss(),
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        training_hooks=[restoreCheckpointHook]
    )


def my_restore_hook():
    checkpoint_path_dict = {'ResNet_50':  '/home/dongkuiyao/model_zoo/resnet_v1_50.ckpt',
                            'VGG_16':  '/home/dongkuiyao/model_zoo/vgg_16.ckpt'
                           }
    var_scopes_dict = {'ResNet_50': 'resnet_v1_50',
                       'VGG_16': 'vgg_16'}
    checkpoint_exclude_scopes_dict = {'ResNet': '',
                                      'VGG_16': 'vgg_16/conv1,vgg_16/pool5,vgg_16/fc6, vgg_16/dropout6, vgg_16/fc8'}
        

    restoreCheckpointHook = CheckpointSaverHook(checkpoint_path_dict,var_scopes_dict, checkpoint_exclude_scopes_dict )
    return restoreCheckpointHook



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    #定义超参数
    model_params = {"learning_rate":0.001}
    #定义训练的相关配置参数
    #keep_checkpoint_max=1表示在只在目录下保存一份模型文件
    #log_step_count_steps=50表示每训练50次输出一次损失的值
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=10,log_step_count_steps=50)

    estimator = tf.estimator.Estimator(model_fn=main_model_fn,params=model_params,model_dir="./path/model",config=run_config)

    record_path  = "/data2/dongkuiyao/matting/tfrecord/train.record"

    estimator.train(input_fn=lambda :input_fn(record_path,True,'WHOLE',2 , 128),steps=100000)   


if __name__ == "__main__":
    main()