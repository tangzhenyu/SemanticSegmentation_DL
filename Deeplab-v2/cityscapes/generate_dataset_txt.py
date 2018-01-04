import os, glob, sys

# Print an error message and quit
def printError(message):
	print('ERROR: {}'.format(message))
	sys.exit(-1)

def main():
	# Where to look for Cityscapes
	cityscapesPath = os.environ['CITYSCAPES_DATASET']
	# how to search for all ground truth
	searchTrainFine = os.path.join(cityscapesPath, "gtFine", "train" , "*", "*_gt*_labelTrainIds.png")
	searchValFine = os.path.join(cityscapesPath, "gtFine", "val" , "*", "*_gt*_labelTrainIds.png")
	searchTrainCoarse = os.path.join(cityscapesPath, "gtCoarse", "train" , "*", "*_gt*_labelTrainIds.png")
	searchValCoarse = os.path.join(cityscapesPath, "gtCoarse", "val" , "*", "*_gt*_labelTrainIds.png")
	searchExTrainCoarse = os.path.join(cityscapesPath, "gtCoarse", "train_extra", "*", "*_gt*_labelTrainIds.png")
	searchTrainImg = os.path.join(cityscapesPath, "leftImg8bit", "train" , "*", "*_leftImg8bit.png")
	searchValImg = os.path.join(cityscapesPath, "leftImg8bit", "val" , "*", "*_leftImg8bit.png")
	searchExTrainImg = os.path.join(cityscapesPath, "leftImg8bit", "train_extra" , "*", "*_leftImg8bit.png")
	searchTestImg = os.path.join(cityscapesPath, "leftImg8bit", "test" , "*", "*_leftImg8bit.png")

	# search files
	filesTrainFine = glob.glob(searchTrainFine)
	filesTrainFine.sort()
	filesValFine = glob.glob(searchValFine)
	filesValFine.sort()
	filesTrainCoarse = glob.glob(searchTrainCoarse)
	filesTrainCoarse.sort()
	filesValCoarse = glob.glob(searchValCoarse)
	filesValCoarse.sort()
	filesExTrainCoarse = glob.glob(searchExTrainCoarse)
	filesExTrainCoarse.sort()
	filesTrainImg = glob.glob(searchTrainImg)
	filesTrainImg.sort()
	filesValImg = glob.glob(searchValImg)
	filesValImg.sort()
	filesExTrainImg = glob.glob(searchExTrainImg)
	filesExTrainImg.sort()
	filesTestImg = glob.glob(searchTestImg)
	filesTestImg.sort()

	# quit if we did not find anything
	if not filesTrainFine:
		printError("Did not find any gtFine/train files.")
	if not filesValFine:
		printError("Did not find any gtFine/val files.")
	if not filesTrainCoarse:
		printError("Did not find any gtCoarse/train files.")
	if not filesValCoarse:
		printError("Did not find any gtCoarse/val files.")
	if not filesExTrainCoarse:
		printError("Did not find any gtCoarse/train_extra files.")
	if not filesTrainImg:
		printError("Did not find any leftImg8bit/train files.")
	if not filesValImg:
		printError("Did not find any leftImg8bit/val files.")
	if not filesExTrainImg:
		printError("Did not find any leftImg8bit/train_extra files.")
	if not filesTestImg:
		printError("Did not find any leftImg8bit/test files.")

	# assertion
	assert len(filesTrainImg) == len(filesTrainFine), \
		"Error %d (filesTrainImg) != %d (filesTrainFine)" % (len(filesTrainImg), len(filesTrainFine))
	assert len(filesTrainImg) == len(filesTrainCoarse), \
		"Error %d (filesTrainImg) != %d (filesTrainCoarse)" % (len(filesTrainImg), len(filesTrainCoarse))
	assert len(filesValImg) == len(filesValFine), \
		"Error %d (filesValImg) != %d (filesValFine)" % (len(filesValImg), len(filesValFine))
	assert len(filesValImg) == len(filesValCoarse), \
		"Error %d (filesValImg) != %d (filesValCoarse)" % (len(filesValImg), len(filesValCoarse))
	assert len(filesExTrainImg) == len(filesExTrainCoarse), \
		"Error %d (filesExTrainImg) != %d (filesExTrainCoarse)" % (len(filesExTrainImg), len(filesExTrainCoarse))
	assert len(filesTestImg) == 1525, "Error %d (filesTestImg) != 1525" % len(filesTestImg)
	files = filesTrainFine+filesValFine+filesTrainCoarse+filesValCoarse+filesExTrainCoarse
	assert len(files) == 26948, "Error %d (gtFiles) != 26948" % len(files)

	# create txt
	dir_path = os.path.join(cityscapesPath, 'dataset')
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	print("---create test.txt---")
	with open(os.path.join(dir_path, 'test.txt'), 'w') as f:
		for l in filesTestImg:
			f.write(l[len(cityscapesPath):] + '\n')
	print("---create train_fine.txt---")
	with open(os.path.join(dir_path, 'train_fine.txt'), 'w') as f:
		for l in zip(filesTrainImg, filesTrainFine):
			assert l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')] \
				== l[1][len('/tempspace2/zwang6/Cityscapes/gtFine/'):-len('_gtFine_labelTrainIds.png')], \
				"%s != %s" % (l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')], \
				l[1][len('/tempspace2/zwang6/Cityscapes/gtFine/'):-len('_gtFine_labelTrainIds.png')])
			f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')
	print("---create val_fine.txt---")
	with open(os.path.join(dir_path, 'val_fine.txt'), 'w') as f:
		for l in zip(filesValImg, filesValFine):
			assert l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')] \
				== l[1][len('/tempspace2/zwang6/Cityscapes/gtFine/'):-len('_gtFine_labelTrainIds.png')], \
				"%s != %s" % (l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')], \
				l[1][len('/tempspace2/zwang6/Cityscapes/gtFine/'):-len('_gtFine_labelTrainIds.png')])
			f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')
	print("---create train_coarse.txt---")
	with open(os.path.join(dir_path, 'train_coarse.txt'), 'w') as f:
		for l in zip(filesTrainImg, filesTrainCoarse):
			assert l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')] \
				== l[1][len('/tempspace2/zwang6/Cityscapes/gtCoarse/'):-len('_gtCoarse_labelTrainIds.png')], \
				"%s != %s" % (l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')], \
				l[1][len('/tempspace2/zwang6/Cityscapes/gtCoarse/'):-len('_gtCoarse_labelTrainIds.png')])
			f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')
	print("---create val_coarse.txt---")
	with open(os.path.join(dir_path, 'val_coarse.txt'), 'w') as f:
		for l in zip(filesValImg, filesValCoarse):
			assert l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')] \
				== l[1][len('/tempspace2/zwang6/Cityscapes/gtCoarse/'):-len('_gtCoarse_labelTrainIds.png')], \
				"%s != %s" % (l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')], \
				l[1][len('/tempspace2/zwang6/Cityscapes/gtCoarse/'):-len('_gtCoarse_labelTrainIds.png')])
			f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')
	print("---create train_extra.txt---")
	with open(os.path.join(dir_path, 'train_extra.txt'), 'w') as f:
		for l in zip(filesExTrainImg, filesExTrainCoarse):
			assert l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')] \
				== l[1][len('/tempspace2/zwang6/Cityscapes/gtCoarse/'):-len('_gtCoarse_labelTrainIds.png')], \
				"%s != %s" % (l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')], \
				l[1][len('/tempspace2/zwang6/Cityscapes/gtCoarse/'):-len('_gtCoarse_labelTrainIds.png')])
			f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')
	print("---create train.txt---")
	with open(os.path.join(dir_path, 'train.txt'), 'w') as f:
		for l in zip(filesTrainImg+filesExTrainImg, filesTrainFine+filesExTrainCoarse):
			# rough match: len('gtCoarse') > len('gtFine')
			assert l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')] \
				== l[1][len('/tempspace2/zwang6/Cityscapes/gtCoarse/'):-len('_gtCoarse_labelTrainIds.png')] \
				or l[0][len('/tempspace2/zwang6/Cityscapes/leftImg8bit/'):-len('_leftImg8bit.png')] \
				== l[1][len('/tempspace2/zwang6/Cityscapes/gtFine/'):-len('_gtFine_labelTrainIds.png')], \
				"%s != %s" % (l[0], l[1])
			f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')

# call the main
if __name__ == "__main__":
	os.environ['CITYSCAPES_DATASET'] = '/tempspace2/zwang6/Cityscapes'
	main()