from PIL import Image, ImageDraw
import scipy.ndimage
import scipy.io
import numpy as np
import time
import copy


class BaseDraw:
	def __init__(self, color150, objectNames, img, pred_size, predicted_classes):
		self.class_colors = scipy.io.loadmat(color150)
		self.class_names = scipy.io.loadmat(objectNames, struct_as_record=False)
		self.im = img
		self.pred_size = pred_size
		self.predicted_classes = copy.deepcopy(predicted_classes)
		self.original_W = self.im.size[0]
		self.original_H = self.im.size[1]

		self.output_W = self.original_W
		self.output_H = self.original_H	


	def dumpArray(self, array, i):
		test = array*100
		test = Image.fromarray(test.astype('uint8'))
		test = test.convert("RGB")
		test.save('/home/vlad/oS_AI/'+str(i)+'t.jpg', "JPEG")

	def calculateResize(self):
		W_coef = float(self.original_W)/float(self.output_W)
		H_coef = float(self.original_H)/float(self.output_H)
		horiz_pad = 0
		vert_pad = 0
		if W_coef > H_coef:
			coef = W_coef
			horiz_pad = int((self.output_H - self.original_H/coef)/2)
			return [coef, horiz_pad, vert_pad]
		else:
			coef = H_coef
			vert_pad = int((self.output_W - self.original_W/coef)/2)
			return [coef, horiz_pad, vert_pad]


	def resizeToOutput(self, image, coef, h_pad, w_pad):
		image = image.resize((int(self.original_W/coef), int(self.original_H/coef)), resample=Image.BILINEAR)
		outputImage = Image.new("RGB",(self.output_W,self.output_H),(0,0,0))
		outputImage.paste(image,(w_pad,h_pad))
		return outputImage



	def drawSimpleSegment(self):

		#Drawing module
		im_Width, im_Height = self.pred_size
		prediction_image = Image.new("RGB", (im_Width, im_Height) ,(0,0,0))
		prediction_imageDraw = ImageDraw.Draw(prediction_image)

		#BASE all image segmentation
		for i in range(im_Width):
			for j in range(im_Height):
				#get matrix element class(0-149)
				px_Class = self.predicted_classes[j][i]
				#assign color from .mat list
				put_Px_Color = tuple(self.class_colors['colors'][px_Class])

				#drawing
				prediction_imageDraw.point((i,j), fill=put_Px_Color)

		#Resize to original size and save
		self.coef, self.h_pad, self.w_pad = self.calculateResize()
		FullHdOutImage = self.resizeToOutput(prediction_image, self.coef, self.h_pad, self.w_pad)
		FullHdOutImage = Image.blend(FullHdOutImage, self.im, 0.5)

		return FullHdOutImage
