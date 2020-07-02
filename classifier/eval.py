# coding:utf-8
# ! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

# ============
# 命令行执行：python eval.py --eval_train --checkpoint_dir="./runs/1541671904/checkpoints/"
# 预测结果在runs/1541671904/prediction.csv中。
# ============

# 参数：我们这里使用命令行传入参数的方式执行该脚本
tf.flags.DEFINE_string("positive_data_file", "./fenci/T_test1263.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./fenci/F_test1274.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 填写训练获得模型的存储位置
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1592381154/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    # x_raw = ["a masterpiece four years in the making", "everything is off."]
    # y_test = [1, 0]
    x1 =''
    x_raw = [x1]
    y_test = [1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print("checkpoint_file========", checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #加载模型
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

        saver.restore(sess, checkpoint_file)

        # 按名称从图中获取占位符
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # 计算张量
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # 计算每一个迭代的批次
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # 存储模型预测结果
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
    r = recall_score(y_test, all_predictions, average='binary')
    print("Recall:", r)

# [0,1]
# [1,0]

# 将二分类的0，1标签转化为中文标签
y = []
for i in all_predictions:
    if i == 0.0:
        y.append("[其他]")
    else:
        y.append("[美国实体清单]")
# 把预测的结果保存到本地
predictions_human_readable = np.column_stack((y, np.array(x_raw)))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction2.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w',newline='') as f:
    csv.writer(f).writerows(predictions_human_readable)
