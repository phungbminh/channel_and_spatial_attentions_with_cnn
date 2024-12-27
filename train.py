import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, BackupAndRestore, LambdaCallback
import keras
import argparse
import pickle
import wandb
from wandb.integration.keras import WandbMetricsLogger

import os, sys, argparse, pytz, json
from datetime import datetime
from model_cnn import resnet50,vgg16, resnet18
from model_cnn_v2 import ResNet, VGG16
from tensorflow.keras import layers, Model


def main():
    root_dir = "kaggle/working"

    current_time = datetime.now()

    # Convert to Vietnam time zone
    vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time_vietnam = current_time.astimezone(vietnam_timezone)

    # Format the datetime object to exclude milliseconds
    current_time = str(current_time_vietnam.strftime('%Y-%m-%d %H:%M:%S'))

    current_time = current_time.replace(" ", "_")
    current_time = current_time.replace("-", "_")
    current_time = current_time.replace(":", "_")
    current_time = current_time.replace("+", "_")

    parser = argparse.ArgumentParser()

    # Arguments users used when running command lines
    parser.add_argument('--result-path', type=str, default="./working", metavar='RESULT_DIR', help='')
    parser.add_argument('--train-folder', default='/kaggle/input/fer2013/train', type=str,  help='Where training data is located')
    parser.add_argument('--valid-folder', default='/kaggle/input/fer2013/test', type=str,  help='Where validation data is located')
    parser.add_argument('--model', default='vgg16', type=str, help='Type of model')
    parser.add_argument('--num-classes', default=7, type=int, help='Number of classes')
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument('--image-size', default=48, type=int, help='Size of input image')
    parser.add_argument('--optimizer', default='adamax', type=str, help='Types of optimizers')
    parser.add_argument('--lr-scheduler', default='ExponentialDecay', type=str, help='Types of scheduler')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=120, type=int, help='Number of epochs')
    parser.add_argument('--image-channels', default=1, type=int, help='Number channel of input image')
    parser.add_argument('--class-mode', default='sparse', type=str, help='Class mode to compile')
    parser.add_argument('--model-path', default=current_time + '.h5.keras', type=str, help='Path to save trained model')
    parser.add_argument('--class-names-path', default='class_names.pkl', type=str, help='Path to save class names')
    parser.add_argument('--early-stopping', default=50, type=str, help='early stopping for avoiding overfit')
    parser.add_argument('--d-steps', default=32, type=int, help='step per epochs')
    parser.add_argument('--use-wandb', default=1, type=int, help='Use wandb')
    parser.add_argument('--wandb-api-key', default='cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b', type=str, help='wantdb api key')
    parser.add_argument('--wandb-project-name', default='Resnet50_BAM_v2', type=str,help='name project to store data in wantdb')
    parser.add_argument('--wandb-runer', default='', type=str, help='')
    parser.add_argument('--attention_option', default='None', type=str, help='CBAM, BAM, scSE')
    parser.add_argument('--color-mode', default='grayscale', type=str, help='Color mode')


    # args, unknown = parser.parse_known_args()

    args = parser.parse_args()
    print(args)
    experiments_dir = args.result_path + '/' + current_time
    if args.use_wandb == 1:
        wandb.login(key=args.wandb_api_key)
        # Initialize WandB with the configuration from the parsed arguments
        wandb.init(project=args.wandb_project_name,  name=args.wandb_runer, config=vars(args))

    # chuan bi dataset de training
    TRAINING_DIR = args.train_folder
    TEST_DIR = args.valid_folder
    print(TRAINING_DIR)
    print(TEST_DIR)

    loss = SparseCategoricalCrossentropy()
    class_mode = args.class_mode
    classes = args.num_classes

    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    img_size = args.image_size
    print(img_size)
    train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(img_size, img_size), batch_size=args.batch_size, class_mode=class_mode, color_mode=args.color_mode)
    val_generator = val_datagen.flow_from_directory(TEST_DIR, target_size=(img_size, img_size), batch_size=args.batch_size, class_mode=class_mode,color_mode=args.color_mode)
    # Luu Tru lai chi muc nhan
    class_names = list(train_generator.class_indices.keys())
    with open(args.class_names_path, 'wb') as fp:
        pickle.dump(class_names, fp)

    model = Model()
    if args.model == 'resnet50':
        model = resnet50(input_shape=(args.image_size, args.image_size, args.image_channels), num_classes=classes,
                        attention_type=args.attention_option)

    if args.model == 'resnet18':
        # model = resnet18(input_shape=(args.image_size, args.image_size, args.image_channels), num_classes=classes,
        #               attention_type=args.attention_option)
        model = ResNet(model_name="ResNet18", input_shape=(args.image_size, args.image_size, args.image_channels),
                       attention=args.attention_option,  pooling="avg")
    if args.model == 'vgg16':
        model = VGG16(input_shape=(args.image_size, args.image_size, args.image_channels), num_classes=classes,
                      attention_type=args.attention_option)
    model.build(input_shape=(None, args.image_size, args.image_size, args.image_channels))
    model.summary()

    # chon learning rate scheduler
    lr_schedule = args.lr
    if args.lr_scheduler == 'ExponentialDecay':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.lr,
            decay_steps=10000,
            decay_rate=0.9)
    elif args.lr_scheduler == 'CosineDecay':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=5,
            alpha=0.0,
            name="CosineDecay",
            warmup_target=None,
            warmup_steps=0)

    # chon bo toi uu optimizer
    if (args.optimizer == 'adam'):
        optimizer = Adam(learning_rate=lr_schedule)
    elif (args.optimizer == 'sgd'):
        optimizer = SGD(learning_rate=lr_schedule)
    elif (args.optimizer == 'rmsprop'):
        optimizer = RMSprop(learning_rate=lr_schedule)
    elif (args.optimizer == 'adadelta'):
        optimizer = Adadelta(learning_rate=lr_schedule)
    elif (args.optimizer == 'adamax'):
        optimizer = Adamax(learning_rate=lr_schedule)
    else:
        raise 'Invalid optimizer. Valid option: adam, sgd, rmsprop, adadelta, adamax'

    callbacks = []
    # wandb
    if args.use_wandb == 1:
        cb_wandb = WandbMetricsLogger(log_freq=1)
        callbacks.append(cb_wandb)

    # logger
    log_path = experiments_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cb_log = CSVLogger(log_path + '/log.csv')
    callbacks.append(cb_log)

    # early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.early_stopping))

    # Thiet lap ham loss
    model.compile(optimizer=optimizer,
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Luu lai he so weight
    best_model = ModelCheckpoint(args.model_path,
                                 save_weights_only=False,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True)
    callbacks.append(best_model)

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        verbose=1,
        validation_data=val_generator,
        callbacks=callbacks,
    )

    best_val_accuracy = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])

    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    # 🐝 Close your wandb run
    if args.use_wandb == 1:
        wandb.finish()



if __name__ == "__main__":
    main()
