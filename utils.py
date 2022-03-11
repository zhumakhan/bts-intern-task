import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TeethDataset, TestDataset
from loss import dice_metric

def save_checkpoint(state, filename="checkpoint.pth.tar", epoch=0):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def load_model(model,filename):
	model.load_state_dict(torch.load(filename))

def get_loaders(
		image_dir,
		mask_dir,
		val_percentage=0.1,
		batch_size=4,
		image_transform=None,
		mask_transform=None,
		num_workers=2,
		pin_memory=False,
	):
	
	dataset = TeethDataset(image_dir=image_dir, mask_dir=mask_dir, image_transform=image_transform, mask_transform=mask_transform)
	
	val_size = int(val_percentage * len(dataset))
	train_size = len(dataset) - val_size

	train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(
		train_data,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=True,
	)
	val_loader = DataLoader(
		val_data,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=True,
	)

	return train_loader, val_loader


def get_test_loader(image_dir,batch_size=4,image_transform=None,num_workers=2,pin_memory=False):
	dataset = TestDataset(image_dir, image_transform)

	return DataLoader(
		dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=False
	)
	

def check_accuracy(loader, model, device='cpu'):
	num_correct = 0
	num_pixels = 0
	dice_score = 0

	model.eval()

	with torch.no_grad():
		for x,y in loader:
			x = x.to(device)
			y = y.to(device).unsqueeze(1)
			preds = torch.sigmoid(model(x))
			preds = (preds > 0.5).float()
			num_correct += (preds == y).sum()
			num_pixels += torch.numel(preds)
			dice_score += dice_metric(preds, y)

	print(f'Got {num_correct/num_pixels*100}')
	print(f'Dice score {dice_score/len(loader)}')
	model.train()

def save_predictions_as_imgs(loader, model, folder='outputs/', device='cpu'):
	model.eval()
	for idx, (x,y) in enumerate(loader):
		x = x.to(device=device)
		with torch.no_grad():
			preds = torch.sigmoid(model(x))
			preds = (preds > 0.5).float()

		torchvision.utils.save_image(preds, f'{folder}Image{idx}.png')

	model.train()



def save_predictions_test(loader, model, folder='outputs/', device='cpu'):
	model.eval()
	for idx, (x,y) in enumerate(loader):
		x = x.to(device=device)
		with torch.no_grad():
			preds = torch.sigmoid(model(x))
			preds = (preds > 0.5).float()

		torchvision.utils.save_image(preds, f'{folder}{y[0][:-4]}.png')

	model.train()