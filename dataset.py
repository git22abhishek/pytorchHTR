from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize

from PIL import Image
from PIL import ImageFile
import pandas as pd
import glob

from xml.etree import ElementTree
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True


class IAM(Dataset):

	def __init__(self, root_dir, transforms=None):

		self.root_dir = root_dir
		self.transforms = transforms
		self.data = self._create_df()
		self.charset = self.get_charset()

	def __getitem__(self, index):

		image_name = self.data.at[index, 'Image']
		image = self._read_image(image_name)
		
		if self.transforms is not None:
			image = self.transforms(image)
		else:
			image = image.rotate(-90, expand=True)
			transforms = Compose([
				Resize((1024, 128)),
				ToTensor(),
				])
			image = transforms(image)

		target = self.data.at[index, 'Transcription']

		return (image, target)


	def __len__(self):
		return len(self.data)


	def _read_image(self, image_name):

		path = image_name.split('-') # ['a01', '000u', '00.png']
		path = os.path.join(
			self.root_dir,
			'lines',
			path[0],
			'-'.join(path[:2]),
			image_name
			)

		return Image.open(path).convert("L") # Convert to Grayscale


	def _create_df(self):

		col_names = ['Image', 'Segmentation', 'Transcription', 'Threshold']
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
				threshold = line.attrib['threshold'] # threshold for binarization

				rows.append({
					'Image': line_id + '.png', 
					'Segmentation': segmentation, 
					'Transcription': transcription,
					'Threshold': threshold,
					})
			
		return pd.DataFrame(rows, columns=col_names)


	def get_charset(self):

		data = self.data
		chars = []

		data.Transcription.apply(lambda x: chars.extend(list(x)))
		chars = ''.join(sorted(set(chars)))

		return chars