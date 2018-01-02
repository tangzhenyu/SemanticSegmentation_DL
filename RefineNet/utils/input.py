# coding:utf-8
import glob
import cv2
from PIL import  Image
import time
import os
import numpy as np
import tensorflow as tf
from input_utils import GeneratorEnqueuer
from utils.aug import resize_image_with_annotation

FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(FLAGS.training_data_path+'SegmentationClass/', '*.{}'.format(ext))))
    return files

def generator(batch_size=12):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        segs = []
        for i in index:
            try:
                seg_fn=image_list[i]
                seg=np.expand_dims(np.array(Image.open(seg_fn)),axis=-1)

                base_name=seg_fn.split('/')[-1].split('.')[0]
                im_fn=os.path.join(FLAGS.training_data_path,'JPEGImages/',base_name+'.jpg')
                im=cv2.imread(im_fn)

                resized_img, resized_seg=resize_image_with_annotation(im, seg, (384,384))


                images.append(resized_img[:, :, ::-1].astype(np.float32))
                segs.append(resized_seg[:,:,::-1].astype(np.float32))

                if len(images) == batch_size:
                    images=np.array(images)
                    segs=np.array(segs)
                    yield images,segs
                    images = []
                    segs=[]
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



if __name__ == '__main__':
    generator(12)
