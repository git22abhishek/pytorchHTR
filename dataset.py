from torch.utils.data import Dataset
import pandas as pd
import cv2
import glob

from xml.etree import ElementTree
import os

class IAM(Dataset):

	def __init__(self, root_dir, split, transforms=None):

		assert split in ('train', 'val', 'test')

		self.root_dir = root_dir
		self.split = split
		self.transforms = transforms
		self.charset = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		self.data = self._create_split()

	def __getitem__(self, index):

		image_name = self.data.at[index, 'Image']
		image = self._read_image(image_name)
		
		if self.transforms:
			image = self.transforms(image)

		target = self.data.at[index, 'Transcription']

		return (image, target)


	def __len__(self):
		return len(self.data)


	def _create_split(self):

		def get_form_name(line_id):
			return '-'.join(line_id.split('-')[:2])

		col_names = ['Image', 'Segmentation', 'Transcription']
		rows = []

		path = os.path.join(self.root_dir, 'largeWriterIndependentTextLineRecognitionTask')

		if self.split == 'val':
			lines = []
			for file in ['validationset1.txt', 'validationset2.txt']:
				ids_path = os.path.join(path, file)
				lines += open(ids_path).read().splitlines()
			pass
		else:
			ids_path = os.path.join(path, f"{self.split}set.txt")
			lines = open(ids_path).read().splitlines()

		forms = list(set(map(get_form_name, lines)))

		for form in forms:
			xml_file = os.path.join(self.root_dir, 'xml', f"{form}.xml")

			# Parse the xml file
			dom = ElementTree.parse(xml_file)
			root = dom.getroot()

			# Iterate through all the lines in a form
			for line in root.iter('line'):

				transcription = line.attrib['text'].replace('&quot;', '"')
				segmentation = line.attrib['segmentation'] # result of segmentation, either 'ok' or 'err'
				line_id = line.attrib['id']

				rows.append({
					'Image': line_id + '.png', 
					'Segmentation': segmentation, 
					'Transcription': transcription
					})
			
		return pd.DataFrame(rows, columns=col_names)


	def _read_image(self, image_name):

		path = image_name.split('-') # ['a01', '000u', '00.png']
		path = os.path.join(
			self.root_dir,
			'lines',
			path[0],
			'-'.join(path[:2]),
			image_name
			)

		return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


	def get_entire_dataset(self):

		col_names = ['Image', 'Segmentation', 'Transcription']
		rows = []

		xml_files = sorted(glob.glob(os.path.join(self.root_dir, 'xml', '*.xml')))

		for xml_file in xml_files:
			form = os.path.join(self.root_dir, 'xml', xml_file)

			# Parse the xml file
			dom = ElementTree.parse(form)
			root = dom.getroot()

			# Iterate through all the lines in a form
			for line in root.iter('line'):

				transcription = line.attrib['text'].replace('&quot;', '"')
				segmentation = line.attrib['segmentation'] # result of segmentation, either 'ok' or 'err'
				line_id = line.attrib['id']

				rows.append({
					'Image': line_id + '.png', 
					'Segmentation': segmentation, 
					'Transcription': transcription
					})
			
		return pd.DataFrame(rows, columns=col_names)


# meow = IAM('/mnt/d/Machine-Learning/Datasets/iamdataset/uncompressed', split='test')

