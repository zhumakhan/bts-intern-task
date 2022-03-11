import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


from model import Unet
from loss import dice_loss, bce_dice_loss


from utils import(
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	check_accuracy,
	save_predictions_as_imgs,
	save_predictions_test,
	get_test_loader,
	load_model,
)


LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NUM_EPOCHS = 2
NUM_WORKERS = 2
# IMAGE_HEIGHT = 768
# IMAGE_WIDTH = 576
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
PIN_MEMORY = True
LOAD_MODEL = False

IMG_DIR = 'dataset/sobel/imgs/'
MASK_DIR = 'dataset/masks/'

TEST_DIR = 'dataset/test/imgs/'
TEST_OUT_DIR = 'dataset/test/masks/'
MODEL_FILE = 'model.pkl'

def train(loader, model, optimizer, loss_fn, scaler):
	loop = tqdm(loader)

	for batch_idx, (data, targets) in enumerate(loop):
		data = data.to(device=DEVICE)
		targets = targets.float().unsqueeze(1).to(device=DEVICE)

		# forward

		# with torch.cuda.amp.autocast():
		with torch.autocast(device_type=DEVICE):
			preds = model(data)
			loss = loss_fn(preds, targets)

		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# scaler.scale(loss).backward()
		# scaler.step(optimizer)
		# scaler.update()

		# update tqdm loop
		loop.set_postfix(loss=loss.item())





def main():
	image_transform = A.Compose(
		[
			A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
			A.Normalize(
				mean=[83/255, 88/255, 93/255],
				std=[0.105, 0.107, 0.108],
				max_pixel_value=1.0,
			),
			ToTensorV2(),
		],
	)
	mask_transform = A.Compose(
		[
			A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
			ToTensorV2(),
		],
	)

	model = Unet(in_channels=3, out_channels=1).to(device=DEVICE)
	
	loss_fn = nn.BCEWithLogitsLoss()
	# loss_fn = dice_loss
	# loss_fn = bce_dice_loss
	
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	scaler = torch.cuda.amp.GradScaler()
	

	train_loader, val_loader = get_loaders(
		image_dir=IMG_DIR,
		mask_dir=MASK_DIR,
		val_percentage=0.1,
		batch_size=BATCH_SIZE,
		image_transform=image_transform,
		mask_transform=mask_transform,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
	)


	for epoch in range(NUM_EPOCHS):
		train(train_loader, model, optimizer, loss_fn, scaler)

		# save model
		# checkpoint = {
		# 	'state_dict':model.state_dict(),
		# 	'optimizer':optimizer.state_dict()
		# }
		# save_checkpoint(checkpoint)

		# check accuracy
		check_accuracy(val_loader, model, device=DEVICE)

		# print examples
		save_predictions_as_imgs(val_loader, model, folder='dataset/outputs/', device=DEVICE)
	# save model
	save_checkpoint(model.state_dict(), filename=MODEL_FILE)

def predict(image_dir):
	image_transform = A.Compose(
		[
			A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
			A.Normalize(
				mean=[83/255, 88/255, 93/255],
				std=[0.105, 0.107, 0.108],
				max_pixel_value=1.0,
			),
			ToTensorV2(),
		],
	)

	model = Unet(in_channels=3, out_channels=1).to(device=DEVICE)

	load_model(model, filename=MODEL_FILE)

	test_loader = get_test_loader(
		image_dir=image_dir,
		batch_size=BATCH_SIZE,
		image_transform=image_transform,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
	)

	save_predictions_test(test_loader, model, folder=TEST_OUT_DIR, device=DEVICE)

if __name__ == '__main__':
	main()
	# predict(TEST_DIR)











