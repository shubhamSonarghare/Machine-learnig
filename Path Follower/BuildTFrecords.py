'''
Run with command:
python BuildTFrecords.py -train_dir = "Name_of_train_dir" - validation_dir = "validation_dir_name" -tfrecord_filename = "Record_filename"

'''
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib
from PIL import Image
import random
from skimage.data import imread

flags = tf.app.flags

flags.DEFINE_string('train_dir', None,
                           'Training data directory')
flags.DEFINE_string('validation_dir', None,
							'Validation data directory')

flags.DEFINE_string('tfrecord_filename', None,
                           'Output tfrecord filename')

FLAGS = flags.FLAGS

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def writeTFrecord(img_lst, label_lst, filename, record_type):
	tfrecords_filename = filename+"_"+record_type+".tfrecord"
	print("Writing " + record_type + " data to current directory")
	writer = tf.python_io.TFRecordWriter(tfrecords_filename)
	ln = len(img_lst)
	cnt=0
	for img_path in img_lst:
		img = np.array(Image.open(img_path))
		label, label_txt = [(idx+1,z) for idx, z in enumerate(label_lst) if z in img_path][0]
		height = img.shape[0]
		width = img.shape[1]
		#img_raw = img.tostring()
		img_raw = open(img_path, 'rb').read()

		example = tf.train.Example(features=tf.train.Features(feature={
					'image/height': _int64_feature(height),
					'image/width': _int64_feature(width),
					'image/image_raw': _bytes_feature(img_raw),
					'image/class/label': _int64_feature(label),
					'image/class/text': _bytes_feature(tf.compat.as_bytes(label_txt))}))
		writer.write(example.SerializeToString())
		cnt+=1
		print ("["+str(cnt)+"/"+str(ln)+"]",end="\r")
	writer.close()


def process_dataset(data_path, tfrecord_filename, record_type):
	
	print("Getting labels from "+ record_type+ " directory")
	label_lst= [x[0].replace(data_path,'') for x in os.walk(data_path)][1:]
	
	print("Got Labels!!!")
	print (label_lst)

	print("Getting image list from "+record_type +" directories")

	image_lst = []
	for i in label_lst:
		curr_img_path = data_path+i+"/"
		print (curr_img_path)
		for filename in os.listdir(curr_img_path):
			image_lst.append(curr_img_path+filename)
		print(i+"-->"+ str(len(image_lst)))

	print (" TOTAL DETECTED IMAGES:: ",len(image_lst))
	print(image_lst[0])
	random.shuffle(image_lst)
	print (image_lst[0])
	input("Press to continue")
	print (" TOTAL DETECTED IMAGES:: ",len(image_lst))
	writeTFrecord(image_lst,label_lst, tfrecord_filename, record_type)


def main():

	if not FLAGS.tfrecord_filename:
		raise ValueError('Cannot find tfrecord_filename. Please provide the same.')

	if not FLAGS.train_dir:
		raise ValueError('Cannot find train_directory. Please provide the same.')

	training_path = FLAGS.train_dir
	#print (training_path)
	if training_path[-1]!='/':
		training_path+='/'
	
	validation_path = FLAGS.validation_dir
	if validation_path[-1]!='/' and validation_path!=None:
		validation_path+='/'

	process_dataset(training_path, FLAGS.tfrecord_filename, record_type= "train")
	if validation_path != None:
		process_dataset(validation_path, FLAGS.tfrecord_filename, record_type = "validation")



if __name__ == '__main__':
	#tf.app.run()
	main()
