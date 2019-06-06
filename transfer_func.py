import tensorflow as tf
import cv2
import numpy as np
import CNN_model

# debug parameter
def transfer(input_image, CNN_path):

    input_image = np.array(input_image) / 255.0
    tf.reset_default_graph()
    with tf.Session() as sess:
        batch_shape = (1,) + input_image.shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = CNN_model.net(img_placeholder)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(CNN_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('No such CNN_path')

        input_image_4d = input_image[np.newaxis, ...]
        output = sess.run(preds, feed_dict={img_placeholder: input_image_4d})
        out_img = np.clip(output[0], 0, 255).astype(np.int)
        return out_img[0:input_image.shape[0], 0:input_image.shape[1],...]
