# -*- coding: utf-8 -*-
"""DebertaV3_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14kU9jRkzs4jufsFpMoALWDiL7FU1Z_I6

Install Libraries
"""

!pip install -q keras-core --upgrade
!pip install -q keras-nlp --upgrade
!pip install --upgrade -q wandb git+https://github.com/soumik12345/wandb-addons

import os
os.environ["KERAS_BACKEND"] = "torch"  # "jax" or "tensorflow" or "torch"
# os.environ["WANDB_SILENT"] = "false" # for wandb

import keras_nlp
import keras_core as keras
import keras_core.backend as K


import torch
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

cmap = mpl.cm.get_cmap('coolwarm')

print("TensorFlow:", tf.__version__)
# print("JAX:", jax.__version__)
print("Keras:", keras.__version__)
print("KerasNLP:", keras_nlp.__version__)

"""Configuration"""

class CFG:
    verbose = 0  # Verbosity

    wandb = True  # Weights & Biases logging
    competition = 'llm-detect-ai-generated-text'  # Competition name
    _wandb_kernel = 'awsaf49'  # WandB kernel
    comment = 'DebertaV3-MaxSeq_200-ext_s-torch'  # Comment description

    preset = "deberta_v3_base_en"  # Name of pretrained models
    sequence_length = 200  # Input sequence length

    device = 'TPU'  # Device

    seed = 42  # Random seed

    num_folds = 5  # Total folds
    selected_folds = [1]  # Folds to train on

    epochs = 2 # Training epochs
    batch_size = 3  # Batch size
    drop_remainder = True  # Drop incomplete batches
    cache = True # Caches data after one iteration, use only with `TPU` to avoid OOM

    scheduler = 'cosine'  # Learning rate scheduler

    class_names = ["real", "fake"]  # Class names [A, B, C, D, E]
    num_classes = len(class_names)  # Number of classes
    class_labels = list(range(num_classes))  # Class labels [0, 1, 2, 3, 4]
    label2name = dict(zip(class_labels, class_names))  # Label to class name mapping
    name2label = {v: k for k, v in label2name.items()}  # Class name to label mapping

keras.utils.set_random_seed(CFG.seed)

"""Use GPU or TPU to accelerate  """

def get_device():
    "Detect and intializes GPU/TPU automatically"
    try:
        # Connect to TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        # Set TPU strategy
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f'> Running on TPU', tpu.master(), end=' | ')
        print('Num of TPUs: ', strategy.num_replicas_in_sync)
        device=CFG.device
    except:
        # If TPU is not available, detect GPUs
        gpus = tf.config.list_logical_devices('GPU')
        ngpu = len(gpus)
         # Check number of GPUs
        if ngpu:
            # Set GPU strategy
            strategy = tf.distribute.MirroredStrategy(gpus) # single-GPU or multi-GPU
            # Print GPU details
            print("> Running on GPU", end=' | ')
            print("Num of GPUs: ", ngpu)
            device='GPU'
        else:
            # If no GPUs are available, use CPU
            print("> Running on CPU")
            strategy = tf.distribute.get_strategy()
            device='CPU'
    return strategy, device

# Initialize GPU/TPU/TPU-VM
strategy, CFG.device = get_device()
CFG.replicas = strategy.num_replicas_in_sync

"""Data

"""

from pandas import read_csv
train_dataset = read_csv('data/teset_data.csv')[['text', 'label']].reset_index(drop=True)

train_dataset

train_dataset.text = train_dataset.text.str.replace('\n', ' ')
class_0 = train_dataset[train_dataset['label'] == 0]
class_1 = train_dataset[train_dataset['label'] == 1]

# 确定两个类别的样本数量
num_class_0 = len(class_0)
num_class_1 = len(class_1)

# 计算较小的样本数量
min_num = min(num_class_0, num_class_1)

# 对样本较多的类别进行随机抽样
if num_class_0 > min_num:
    class_0 = class_0.sample(n=min_num, random_state=42)  # random_state 确保结果的可复现性
elif num_class_1 > min_num:
    class_1 = class_1.sample(n=min_num, random_state=42)

# 合并数据集
balanced_train_dataset = pd.concat([class_0, class_1])

# 如果需要，可以打乱数据集的顺序
balanced_train_dataset = balanced_train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
train_dataset.head()

# Load external data
train_dataset = balanced_train_dataset

train_dataset['name'] = train_dataset.label.map(CFG.label2name)

# Display information about the external data
print("# External Data: {:,}".format(len(train_dataset)))
print("# Sample:")
train_dataset.head(2)

# Show distribution of answers using a bar plot
plt.figure(figsize=(8, 4))
train_dataset.name.value_counts().plot.bar(color=[cmap(0.0), cmap(0.65)])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Answer distribution for External Data")
plt.show()

df = train_dataset
df['source'] = 'default_score_value'  # 将'score'列的所有值设置为默认字符串
df['id'] = 'default_id_value'
df.head()

from sklearn.model_selection import StratifiedKFold  # Import package

skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)  # Initialize K-Fold

df = df.reset_index(drop=True)  # Reset dataframe index

df['stratify'] = df.label.astype(str)+df.source.astype(str)

df["fold"] = -1  # New 'fold' column

# Assign folds using StratifiedKFold
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify'])):
    df.loc[val_idx, 'fold'] = fold

# Display label distribution for each fold
df.groupby(["fold", "name", "source"]).size()

"""Preprocessing

"""

preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
    preset=CFG.preset, # Name of the model
    sequence_length=CFG.sequence_length, # Max sequence length, will be padded if shorter
)

inp = preprocessor(df.text.iloc[0])  # Process text for the first row

# Display the shape of each processed output
for k, v in inp.items():
    print(k, ":", v.shape)

def preprocess_fn(text, label=None):
    text = preprocessor(text)  # Preprocess text
    return (text, label) if label is not None else text  # Return processed text and label if available

"""DataLoader

"""

def build_dataset(texts, labels=None, batch_size=32,
                  cache=False, drop_remainder=True,
                  repeat=False, shuffle=1024):
    AUTO = tf.data.AUTOTUNE  # AUTOTUNE option
    slices = (texts,) if labels is None else (texts, labels)  # Create slices
    ds = tf.data.Dataset.from_tensor_slices(slices)  # Create dataset from slices
    ds = ds.cache() if cache else ds  # Cache dataset if enabled
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTO)  # Map preprocessing function
    ds = ds.repeat() if repeat else ds  # Repeat dataset if enabled
    opt = tf.data.Options()  # Create dataset options
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)  # Shuffle dataset if enabled
        opt.experimental_deterministic = False
    ds = ds.with_options(opt)  # Set dataset options
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)  # Batch dataset
    ds = ds.prefetch(AUTO)  # Prefetch next batch
    return ds  # Return the built dataset

def get_datasets(fold):
    train_df = df[df.fold!=fold].sample(frac=1)  # Get training fold data

    train_texts = train_df.text.tolist()  # Extract training texts
    train_labels = train_df.label.tolist()  # Extract training labels

    # Build training dataset
    train_ds = build_dataset(train_texts, train_labels,
                             batch_size=CFG.batch_size*CFG.replicas, cache=CFG.cache,
                             shuffle=True, drop_remainder=True, repeat=True)

    valid_df = df[df.fold==fold].sample(frac=1)  # Get validation fold data
    valid_texts = valid_df.text.tolist()  # Extract validation texts
    valid_labels = valid_df.label.tolist()  # Extract validation labels

    # Build validation dataset
    valid_ds = build_dataset(valid_texts, valid_labels,
                             batch_size=min(CFG.batch_size*CFG.replicas, len(valid_df)), cache=CFG.cache,
                             shuffle=False, drop_remainder=True, repeat=False)

    return (train_ds, train_df), (valid_ds, valid_df)  # Return datasets and dataframes

"""Wandb"""

import wandb  # Import wandb library for experiment tracking
import wandb_addons # Additional wandb utilities

try:
    from kaggle_secrets import UserSecretsClient  # Import UserSecretsClient
    user_secrets = UserSecretsClient()  # Create secrets client instance
    api_key = user_secrets.get_secret("WANDB")  # Get API key from Kaggle secrets
    wandb.login(key=api_key)  # Login to wandb with the API key
    anonymous = None  # Set anonymous mode to None
except:
    anonymous = 'must'  # Set anonymous mode to 'must'
    wandb.login(anonymous=anonymous, relogin=True)  # Login to wandb anonymously and relogin if needed


# Initializes the W&B run with a config file and W&B run settings.
def wandb_init(fold):
    config = {k: v for k, v in dict(vars(CFG)).items() if '__' not in k}  # Create config dictionary
    config.update({"fold": int(fold)})  # Add fold to config
    run = wandb.init(project="llm-fake-text",
                     name=f"fold-{fold}|max_seq-{CFG.sequence_length}|model-{CFG.preset}",
                     config=config,
                     group=CFG.comment,
                     save_code=True)
    return run

# Log best result for error analysis
def log_wandb():
    wandb.log({'best_auc': best_auc, 'best_loss': best_loss, 'best_epoch': best_epoch})

# Fetch W&B callbacks
def get_wb_callbacks(fold):
    wb_metr = wandb.keras.WandbMetricsLogger()
    return [wb_metr]  # Return WandB callbacks

"""LR Schedule"""

import math

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 0.6e-6, 0.5e-6 * batch_size, 0.3e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 1, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

_=get_lr_callback(CFG.batch_size*CFG.replicas, plot=True)

def get_callbacks(fold):
    callbacks = []
    lr_cb = get_lr_callback(CFG.batch_size*CFG.replicas)  # Get lr callback
    ckpt_cb = keras.callbacks.ModelCheckpoint(f'fold{fold}.keras',
                                              monitor='val_auc',
                                              save_best_only=True,
                                              save_weights_only=False,
                                              mode='max')  # Get Model checkpoint callback
    callbacks.extend([lr_cb, ckpt_cb])  # Add lr and checkpoint callbacks

    if CFG.wandb:  # If WandB is enabled
        wb_cbs = get_wb_callbacks(fold)  # Get WandB callbacks
        callbacks.extend(wb_cbs)

    return callbacks  # Return the list of callbacks

"""Model
DebertaV3

"""

def build_model():
    # Create a DebertaV3Classifier model
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        CFG.preset,
        preprocessor=None,
        num_classes=1 # one output per one option, for five options total 5 outputs
    )
    inputs = classifier.input
    logits = classifier(inputs)

    # Compute final output
    outputs = keras.layers.Activation("sigmoid")(logits)
    model = keras.Model(inputs, outputs)

    # Compile the model with optimizer, loss, and metrics
    model.compile(
        optimizer=keras.optimizers.AdamW(5e-6),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            keras.metrics.AUC(name="auc"),
        ],
        jit_compile=True
    )
    return model

# with strategy.scope
model = build_model()

model.summary()

keras.utils.plot_model(model, show_shapes=True)

"""Training"""

for fold in CFG.selected_folds:
    # Initialize Weights and Biases if enabled
    if CFG.wandb:
        run = wandb_init(fold)

    # Get train and validation datasets
    (train_ds, train_df), (valid_ds, valid_df) = get_datasets(fold)

    # Get callback functions for training
    callbacks = get_callbacks(fold)

    # Print training information
    print('#' * 50)
    print(f'\tFold: {fold + 1} | Model: {CFG.preset}\n\tBatch Size: {CFG.batch_size * CFG.replicas} | Scheduler: {CFG.scheduler}')
    print(f'\tNum Train: {len(train_df)} | Num Valid: {len(valid_df)}')
    print('#' * 50)

    # Clear TensorFlow session and build the model within the strategy scope
    K.clear_session()
    with strategy.scope():
        model = build_model()

    # Start training the model
    history = model.fit(
        train_ds,
        epochs=CFG.epochs,
        validation_data=valid_ds,
        callbacks=callbacks,
        steps_per_epoch=int(len(train_df) / CFG.batch_size / CFG.replicas),
    )

    # Find the epoch with the best validation accuracy
    best_epoch = np.argmax(model.history.history['val_auc'])
    best_auc = model.history.history['val_auc'][best_epoch]
    best_loss = model.history.history['val_loss'][best_epoch]

    # Print and display best results
    print(f'\n{"=" * 17} FOLD {fold} RESULTS {"=" * 17}')
    print(f'>>>> BEST Loss  : {best_loss:.3f}\n>>>> BEST AUC   : {best_auc:.3f}\n>>>> BEST Epoch : {best_epoch}')
    print('=' * 50)

    # Log best result on Weights and Biases (wandb) if enabled
    if CFG.wandb:
        log_wandb()  # Log results
        wandb.run.finish()  # Finish the run
#         display(ipd.IFrame(run.url, width=1080, height=720)) # show wandb dashboard
    print("\n\n")

"""Prediction"""

import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 加载另一个测试数据集
another_test_data_path = "/content/real_data/teset_data.csv"
another_test_df = pd.read_csv(another_test_data_path)

# 将测试数据转换为列表
another_test_texts = another_test_df["text"].tolist()
another_test_labels = another_test_df["label"].tolist()  # 假设测试数据集包含真实标签

# 使用之前定义的build_dataset函数构建测试数据集
another_test_ds = build_dataset(another_test_texts, batch_size=CFG.batch_size*CFG.replicas, cache=False, shuffle=False, drop_remainder=False, repeat=False)

# 使用训练好的模型进行预测
another_predictions = model.predict(another_test_ds)

# 如果模型输出是sigmoid激活，则predictions已经是概率
# 如果模型输出是logits，则需要转换为概率，例如使用sigmoid函数
# another_predictions = tf.sigmoid(another_predictions).numpy().flatten()

# 将预测结果转换为二进制标签
binary_predictions = [1 if p > 0.5 else 0 for p in another_predictions.flatten()]

# 计算准确率、F1分数和AUC值
accuracy = accuracy_score(another_test_labels, binary_predictions)


# 打印指标
print(f"Accuracy: {accuracy:.4f}")

import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 加载测试数据集
test_data_path = "/kaggle/input/llm-detect-ai-generated-text/test_essays.csv"
test_df = pd.read_csv(test_data_path)

# 将测试数据转换为列表
test_texts = test_df["text"].tolist()

# 使用之前定义的build_dataset函数构建测试数据集
# 假设 CFG.batch_size 和 CFG.replicas 已经定义
test_ds = build_dataset(test_texts, batch_size=CFG.batch_size*CFG.replicas, cache=False, shuffle=False, drop_remainder=False, repeat=False)

# 使用训练好的模型进行预测
# 假设模型已经加载为 model
predictions = model.predict(test_ds)

# 将预测结果转换为概率
# 如果模型输出是sigmoid激活，则predictions已经是概率
# 如果模型输出是logits，则需要转换为概率，例如使用sigmoid函数
# predictions = tf.sigmoid(predictions).numpy().flatten()

# 创建一个新的DataFrame来存储结果
results_df = pd.DataFrame({
    'id': test_df['id'],
    'generated': predictions.flatten()
})

# 保存结果到CSV文件
output_csv_path = 'submission.csv'
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")

model.save("model")

"""# ✍️ | Reference
* [LLM Science Exam: KerasCore + KerasNLP [TPU]](https://www.kaggle.com/code/awsaf49/llm-science-exam-kerascore-kerasnlp-tpu)
* [Keras NLP](https://keras.io/api/keras_nlp/)
* [Triple Stratified KFold with TFRecords](https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords) by @cdeotte
"""

!rm -r /kaggle/working/wandb
