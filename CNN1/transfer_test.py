import tensorflow as tf
import cv2
import numpy as np
import CNN_model

# debug parameter
CNN_path = ''
input_image = cv2.imread('test.jpg')
output_image = 'output.jpg'


def transfer(input_image, CNN_path, output_image, batchsize=1):
    input_image = cv2.resize(input_image, (256,256))
    input_image = np.array(input_image)
    with tf.Session() as sess:
        batch_shape = (batchsize,) + input_image.shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = CNN_model.net(img_placeholder)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(CNN_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('No such CNN_path')

        input_image = input_image[np.newaxis, ...]
        output = sess.run(preds, feed_dict={img_placeholder: input_image})
        cv2.imwrite(output_image, output[0].astype(np.int))


transfer(input_image, CNN_path, output_image)