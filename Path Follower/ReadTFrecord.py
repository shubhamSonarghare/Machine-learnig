import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


class Generate_from_TFRecord():
	def __init__(self, tfrecord_file):
		self.tfrecord = tfrecord_file

	def extract_image_data(self,batch_size, shuffle):
		def _extract_fn(tfrecord):
			features={
				'image/height': tf.FixedLenFeature([], tf.int64),
				'image/width': tf.FixedLenFeature([], tf.int64),
				'image/image_raw': tf.FixedLenFeature([], tf.string),
				'image/class/label': tf.FixedLenFeature([], tf.int64),
				'image/class/text': tf.FixedLenFeature([], tf.string)}

			sample = tf.parse_single_example(tfrecord, features)

			#image = tf.decode_raw(sample['image/image_raw'],tf.uint8)
			image = tf.image.decode_jpeg(sample['image/image_raw'])
			height = tf.cast(sample['image/height'], tf.int32)
			width = tf.cast(sample['image/width'], tf.int32)
			image = tf.reshape(image,[1,height*width*3])
			#image = tf.reshape(image, [height, width, 3])
			label = tf.cast(sample['image/class/label'], tf.int32)
			label_txt = tf.cast(sample['image/class/text'], tf.string)
			
			return [image, label_txt, label, height, width]

		tfrecord_file = self.tfrecord
		dataset = tf.data.TFRecordDataset([tfrecord_file])
		dataset = dataset.map(_extract_fn)

		if batch_size != None:
			print("Preparing batches.....")
			dataset = dataset.batch(batch_size)
		dataset = dataset.repeat(10)
		if shuffle == True:
			print("Shuffling.....")
			dataset= dataset.shuffle(buffer_size = len(tfrecord_file))
		return dataset

	def generate_data(self, batch_size = None, shuffle = False):
		print("TFRecord file used: ", self.tfrecord)
		dataset = self.extract_image_data(batch_size, shuffle)
		iterator = dataset.make_one_shot_iterator()
		next_batch = iterator.get_next()
		return next_batch
		

	def get_num_examples(self):
		cnt=0
		imge, text, lab, ht, wdth = self.generate_data()
		with tf.Session() as sess:
			sess.run(tf.initializers.global_variables())
			try:
				while(1):
					img_dat, txt, labl, hgth, wth = sess.run([imge, text, lab, ht, wdth])
					cnt+=1
			except:
				pass
		return cnt



'''
cnt=0
		with tf.Session() as sess:
			sess.run(tf.initializers.global_variables())
			try:
				for i in range(4):
				#while (1):
					img_dat, txt, labl, hgth, wth = sess.run([imge, text, lab, ht, wdth])
					#print ("Label: ",txt, labl,"\n Dims: " ,hgth, wth)
					print ("IMG: ", img_dat.shape, type(img_dat))
					Y_label = sess.run(tf.one_hot(labl, depth = 5))
					print ("Label: ", txt, labl, Y_label)
					plt.figure()
					plt.title(txt)
					plt.imshow(img_dat)
					plt.show()
					#sess.run(update)
					cnt+=1
					#print(cnt, end = "\r")

			except:
				pass

#extract_image2("MYFIRST_train.tfrecord")
filenames = [os.path.join(".", 'MYFIRST_train.tfrecord')]
#extrac_image('path_follower_validation.tfrecord')
extrac_image('path_foll_compr_validation.tfrecord')
'''

data = Generate_from_TFRecord('path_foll_compr_reshaped_train.tfrecord')
print("Final Cnt: ",data.get_num_examples())

imge, text, lab, ht, wdth = data.generate_data()

with tf.Session() as sess:
	sess.run(tf.initializers.global_variables())
	try:
		for i in range(4):
		#while (1):
			img_dat, txt, labl, hgth, wth = sess.run([imge, text, lab, ht, wdth])
			#print ("Label: ",txt, labl,"\n Dims: " ,hgth, wth)
			print ("IMG: ", img_dat.shape, type(img_dat), hgth, wth)
			Y_label = sess.run(tf.one_hot(labl, depth = 5))
			print ("Label: ", txt, labl, Y_label)
			#plt.figure()
			#plt.title(txt)
			#plt.imshow(img_dat)
			#plt.show()
			#sess.run(update)
			#cnt+=1
			#print(cnt, end = "\r")

	except:
		pass


		
 