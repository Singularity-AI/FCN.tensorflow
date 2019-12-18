from __future__ import print_function
import os
import sys
import time
import datetime
import argparse
import logging

import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import sklearn.metrics as metrics

from six.moves import xrange
from datetime import datetime

sys.path.append("../")
from data_generator import data_loader_from_h5_percent as dataset


logger = None
def _get_logger(logger_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logger_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if args.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(args, image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    logger.info("setting up vgg initialized conv layers ...")
    
    # import pdb; pdb.set_trace()
    model_data = utils.get_model_data(args.model_dir, args.model_url)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])

    # processed_image = utils.process_image(image, mean_pixel)
    processed_image = image
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)
        image_net['pool5'] = pool5

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if args.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        image_net['conv6'] = conv6
        image_net['relu6'] = relu6
        image_net['relu_dropout6'] = relu_dropout6


        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if args.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        image_net['conv7'] = conv7
        image_net['relu7'] = relu7
        image_net['relu_dropout7'] = relu_dropout7

        # ================attention=====================
        attention_w1 = utils.weight_variable([1, 1, 4096, args.embedding_dims], name="attention_w1")
        attention_b1 = utils.bias_variable([args.embedding_dims], name="attention_b1")
        attention = utils.conv2d_basic(relu_dropout7, attention_w1, attention_b1)
        attention = tf.nn.tanh(attention)
        attention_w2 = utils.weight_variable([1, 1, args.embedding_dims, 1], name="attention_w2")
        attention_b2 = utils.bias_variable([1], name="attention_b2")
        attention = utils.conv2d_basic(attention, attention_w2, attention_b2)
        attention = tf.nn.softmax(attention)
        relu_dropout7_with_attention = relu_dropout7 * attention
        # ================attention=====================

        W8 = utils.weight_variable([1, 1, 4096, args.num_classes], name="W8")
        b8 = utils.bias_variable([args.num_classes], name="b8")
        # conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        conv8 = utils.conv2d_basic(relu_dropout7_with_attention, W8, b8)
        image_net['conv8'] = conv8
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, args.num_classes], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        image_net['conv_t1'] = conv_t1
        image_net['fuse_1'] = fuse_1

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        image_net['conv_t2'] = conv_t2
        image_net['fuse_2'] = fuse_2

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], args.num_classes])
        W_t3 = utils.weight_variable([16, 16, args.num_classes, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([args.num_classes], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        image_net['conv_t3'] = conv_t3
        image_net['annotation_pred'] = annotation_pred
        image_net['annotation_pred_4dims'] = tf.expand_dims(annotation_pred, dim=3)

        # add maxpooling
        logits = utils.max_pool_nxn(conv_t3, args.image_size)
        logits = tf.squeeze(logits)
        prediction = tf.nn.softmax(logits, name='preds')
        image_net['logits'] = logits
        image_net['prediction'] = prediction
        
    return tf.expand_dims(annotation_pred, dim=3), logits, prediction, image_net


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if args.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def generator(args):
    objs, generators, train_nb_samples_per_epoch = \
            dataset.get_instance_datagen(args.train_base_path,
                                         args.labels,
                                         args.labels_map,
                                         args.train_nb_per_class,
                                         args.percent,
                                         is_training=args.is_training,
                                         is_shuffle=args.is_shuffle,
                                         )
    logger.info('========================================================')
    logger.info('train_nb_per_class: {}' .format(args.train_nb_per_class))
    logger.info('========================================================')
    batch_g = dataset.batch_generator(generators, args.batch_size, is_shuffle=args.is_shuffle)
    return batch_g


def _analyze_confusion_matrix(final_confusion_matrix, acc_dic, global_step, num_classes, label_list):
    accuracy_dic = {}
    t_count = 0
    labels_name = label_list
    each_class_total_count = np.sum(final_confusion_matrix, axis=1)
    total_sample_count = np.sum(final_confusion_matrix)
    for label_index in range(num_classes):
        if each_class_total_count[label_index] == 0:
            acc = 0.0
        else:
            acc = final_confusion_matrix[label_index][label_index] / each_class_total_count[label_index]
        accuracy_dic[labels_name[label_index]] = (acc, each_class_total_count[label_index])
        t_count += final_confusion_matrix[label_index][label_index]
    accuracy = t_count / total_sample_count

    acc_dic[global_step] = [accuracy]
    for label_name in labels_name:
        acc_dic[global_step].append(accuracy_dic[label_name][0])


def main(args):
    global logger
    if logger is None:
        logger = _get_logger(args.logger_level)

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, args.image_size, args.image_size, 3], name="input_image")
    label = tf.placeholder(tf.int32, shape=args.batch_size)

    pred_annotation, logits, prediction, image_net = inference(args, image, keep_probability)
    pred_labels = tf.argmax(logits, 1)

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=label,
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if args.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    logger.info("Setting up summary op...")
    summary_op = tf.summary.merge_all()

  
    sess = tf.Session()

    logger.info("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=0)

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside args.logs_dir
    train_writer = tf.summary.FileWriter(args.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(args.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())

    if args.pretrained_model_checkpoint_path:
        logger.info('model path: {}' .format(args.pretrained_model_checkpoint_path))
        saver.restore(sess, args.pretrained_model_checkpoint_path)
        logger.info('{}: Pre-trained model restored from {}' .format(
                datetime.now(), args.pretrained_model_checkpoint_path))
        global_step = int(args.pretrained_model_checkpoint_path.split('/')[-1].split('-')[-1])
        logger.info('Succesfully loaded model from {} at step={}.' .format(
            args.pretrained_model_checkpoint_path, global_step))
    else:
        ckpt = tf.train.get_checkpoint_state(args.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('-')[-1])
            logger.info(ckpt.model_checkpoint_path)
            logger.info("Model restored...")
        else:
            global_step = 0

    final_confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    if args.mode == "train":
        train_batch_generator = generator(args)
        for step in range(global_step, args.max_iteration):
            start_time = time.time()
            batch_images, batch_labels, batch_indexes = next(train_batch_generator)
            feed_dict = {image: batch_images, label: batch_labels, keep_probability: 0.85}

            _, gts, preds, output, train_loss, summary_str = sess.run([train_op, label, pred_labels, logits, loss, loss_summary], feed_dict=feed_dict)
            duration = time.time() - start_time
            final_confusion_matrix += metrics.confusion_matrix(gts, preds, labels=range(args.num_classes))
            acc_dic = {}
            _analyze_confusion_matrix(final_confusion_matrix, acc_dic, step, args.num_classes, args.labels_for_confusion)

            if step % 50 == 0:
                examples_per_sec = args.batch_size / float(duration)
                format_str = "{}: step {}, loss = {:.7} ({:.1} examples/sec; {:.3} sec/batch)" .format(
                    datetime.now(), step, train_loss, examples_per_sec, duration)
                logger.info(format_str)
                train_writer.add_summary(summary_str, step)

            if step % 4000 == 0:
                saver.save(sess, args.logs_dir + "model.ckpt", step)
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op, feed_dict=feed_dict))
                summary.value.add(tag='fcn combination/total_acc', simple_value=acc_dic[step][0])
                logger.info('=================train result==================')
                logger.info('train total_acc:' .format(acc_dic[step][0]))
                labels_name = args.labels_for_confusion
                for index, label_name in enumerate(labels_name):
                    summary.value.add(tag='fcn combination/' + label_name, simple_value=acc_dic[step][index + 1])
                    train_writer.add_summary(summary, step)
                    logger.info('train {}_acc: {}' .format(label_name, acc_dic[step][index + 1]))
                logger.info('train confusion_matrix:')
                logger.info(final_confusion_matrix)
                logger.info('===============================================')
                final_confusion_matrix = np.zeros((args.num_classes, args.num_classes))


    elif args.mode == "visualize":
        pass
        # valid_images, valid_annotations = validation_dataset_reader.get_random_batch(args.batch_size)
        # pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
        #                                             keep_probability: 1.0})
        # valid_annotations = np.squeeze(valid_annotations, axis=3)
        # pred = np.squeeze(pred, axis=3)

        # for step in range(args.batch_size):
        #     utils.save_image(valid_images[step].astype(np.uint8), args.logs_dir, name="inp_" + str(5+step))
        #     utils.save_image(valid_annotations[step].astype(np.uint8), args.logs_dir, name="gt_" + str(5+step))
        #     utils.save_image(pred[step].astype(np.uint8), args.logs_dir, name="pred_" + str(5+step))
        #     logger.info("Saved image: %d" % step)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    logger = _get_logger(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--logger_level", type=str, default=logging.INFO, help="logging level")
    parser.add_argument("--train_base_path", type=str, required=False, help="json base folder path")
    parser.add_argument("--model_url", type=str, 
                         default='http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat', 
                         help="vgg model url")
    parser.add_argument("--model_dir", type=str, default="Model_zoo/", help="Path to vgg model mat")
    parser.add_argument("--data_dir", type=str, required=False, help="path to dataset")
    parser.add_argument("--logs_dir", type=str, required=False, help="path to logs directory")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="numbers of threads for data queue")
    parser.add_argument("--percent", type=float, default=0.2, help="the max step of training")
    parser.add_argument("--embedding_dims", type=int, default=512, help="dims for attention embedding")
    parser.add_argument("--max_iteration", type=int, default=int(1e6 + 1), help="max iteration for training")
    parser.add_argument("--num_classes", type=int, default=2, help="class number")
    parser.add_argument("--image_size", type=int, default=224, help="class number")
    parser.add_argument("--is_training", type=bool, default=True, help="Mode train/ test/ visualize")
    parser.add_argument("--is_shuffle", type=bool, default=False, help="shuffle mode: True/ False")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode: True/ False")
    parser.add_argument("--mode", type=str, default='train', help="Mode train/ test/ visualize")
    parser.add_argument("--pretrained_model_checkpoint_path", type=str, required=False, help="pretrained model")
    args, unparsed = parser.parse_known_args()

    args.labels_map = {
    'high_ad': np.int32(1),
    'mid_ad': np.int32(1),
    'low_ad': np.int32(1),
    'mucinous_ad': np.int32(1),
    'ring_ad': np.int32(1),
    'mixed_ad': np.int32(1),
    'inflammation': np.int32(0),
    'lymphocyte': np.int32(0),
    'fat': np.int32(0),
    'smooth_muscle': np.int32(0),
    'normal_mucosa': np.int32(0),
    'neutrophil': np.int32(0),
    'plasmacyte': np.int32(0),
    'histocyte': np.int32(0),
    'eosnophils': np.int32(0)
    }

    args.labels = {
    'high_ad': 'high_ad',
    'mid_ad': 'mid_ad',
    'low_ad': 'low_ad',
    'mucinous_ad': 'mucinous_ad',
    'ring_ad': 'ring_ad',
    'mixed_ad': 'mixed_ad',
    'inflammation': 'inflammation',
    'lymphocyte': 'lymphocyte',
    'fat': 'fat',
    'smooth_muscle': 'smooth_muscle',
    'normal_mucosa': 'normal_mucosa',
    'neutrophil': 'neutrophil',
    'plasmacyte': 'plasmacyte|histocyte|eosnophils'
    }

    args.labels_for_confusion = ['normal', 'tumor']
    args.train_nb_per_class = [1, 14, 7, 4, 1, 5, 10, 3, 5, 8, 4, 1, 1]
    args.batch_size = sum(args.train_nb_per_class)

    args.train_base_path = "/mnt/disk_share/data/colon/colon_160_224/"
    args.logs_dir = "./logs_colon/20191218_test/"

    main(args)
