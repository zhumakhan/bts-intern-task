import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset

class TeethDataset(Dataset):
	def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.image_transform = image_transform
		self.mask_transform = mask_transform
		self.images = os.listdir(image_dir)
		self.masks = os.listdir(mask_dir)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])
		mask_path = os.path.join(self.mask_dir, self.masks[index])

		image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
		mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
		mask[mask > 0] = 1.0
		
		
		if self.image_transform:
			augmentation = self.image_transform(image=image,mask=mask)
			img_aug = augmentation['image']
		else:
			img_aug = image
		
		if self.mask_transform:
			augmentation = self.mask_transform(image=image,mask=mask)
			mask_aug = augmentation['mask']
		else:
			mask_aug = mask

		return img_aug, mask_aug

class TestDataset(Dataset):
	def __init__(self, image_dir, image_transform=None):
		self.image_dir = image_dir
		self.image_transform = image_transform
		self.images = os.listdir(image_dir)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])

		image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
		
		if self.image_transform:
			augmentation = self.image_transform(image=image)
			img_aug = augmentation['image']
		else:
			img_aug = image

		return img_aug, self.images[index]



def test():
	dataset = TeethDataset('dataset/imgs', 'dataset/masks')
	image, mask = dataset[0]
	print(image.shape)
	print(mask.shape)
	for h in range(mask.shape[0]):
		for w in range(mask.shape[1]):
			print(mask[h][w],end = ' ')
		print()

if __name__ == '__main__':
	test()



