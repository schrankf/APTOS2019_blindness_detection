# import pkgs
from glob import glob
import cv2
import numpy as np

class Images:
	def __init__(self, file_path = None, file_extension = '.png', im_size = 224):
		"""
		TODO
		"""
		self.file_extension = file_extension
		self.file_path      = file_path
		self.file_names 	= []
		self.im_size 	 	= im_size



	def im_load(self, id_names):

		# construct file names
		self.id_names = id_names

		# read images
		images = [cv2.imread(ii) for ii in self.file_path + id_names + self.file_extension]	
		# convert to grayscale
		images = [cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY) for ii in images]
		# resize images
		images = [cv2.resize(ii, (self.im_size , self.im_size )) for ii in images]

		self.data = np.expand_dims(np.array(images).astype('float32') / 255., 4)


	def get_im_from_id(self, id):

		return np.squeeze(self.data[tuple([self.id_names == id])])











