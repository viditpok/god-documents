
import pdb
from pathlib import Path
import os
import json

import numpy as np
import torch

from src.vision.part2_dataset import KittiData
from src.vision.cv2_transforms import ToTensor
from src.vision.part6_transfer_learning import model_and_optimizer
from src.vision.part5_pspnet import PSPNet
from types import SimpleNamespace



ROOT_DIR = os.path.abspath(os.curdir)



def test_KittiData_len() -> None:
	""" Ensure length of dataset is properly generated. This essentially tests make_dataset() """
	split = "train"
	data_root = str(ROOT_DIR + "/kitti")
	dataset = KittiData(split, data_root, transform=ToTensor())

	assert len(dataset) == 259



def test_model_kitti() -> None:
	""" Ensure output shape of the model is 2. This essentially tests model_and_optimizer() """
	psp_model = PSPNet(num_classes=11, pretrained=False)

	args = SimpleNamespace(
		**{
			"classes": 11,
			"zoom_factor": 8,
			"layers": 50,
			"ignore_label": 255,
			"arch": "PSPNet",
			"base_lr": 1e-3,
			"momentum": 0.99,
			"weight_decay": 1e-5,
			"pretrained": False
		}
	)
	model, _ = model_and_optimizer(args, psp_model)
	
	batch_size = 5
	H = 201
	W = 201
	x = torch.rand(batch_size, 3, H, W).type(torch.float32)
	y = torch.ones(batch_size, H, W).type(torch.int64) * 255

	logits, yhat, main_loss, aux_loss = model(x, y)

	assert logits.shape == (batch_size, 2, H, W)

	# make sure that the output shape is correct
	assert yhat.shape == (batch_size, H, W)

	assert isinstance(logits, torch.Tensor)
	assert isinstance(yhat, torch.Tensor)
	assert isinstance(main_loss, torch.Tensor)
	assert isinstance(aux_loss, torch.Tensor)

