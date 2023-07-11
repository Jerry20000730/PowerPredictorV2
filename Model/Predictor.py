import os
import time
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from Data import ImageDataset
from Model import UNet, ResNet
from Log import Logger


class EnergyPredictorV2_image2heightmap():
    def __init__(self,
                 train_src_folder_path : str,
                 test_src_folder_path : str,
                 heightmap_src_folder_path: str,
                 checkpoint_folder_path: str,
                 train_batch_size: int = 16,
                 test_batch_size: int = 16,
                 num_epochs: int = 5,
                 device: str = None):

        if train_src_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'train_src_folder_path' for "
                                 "EnergyPredictorV2_image2heightmap is missing")
        if test_src_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'test_src_folder_path' for "
                                 "EnergyPredictorV2_image2heightmap is missing")
        if heightmap_src_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'heightmap_src_folder_path' for "
                                 "EnergyPredictorV2_image2heightmap is missing")
        if checkpoint_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'checkpoint_folder_path' for "
                                 "EnergyPredictorV2_image2heightmap is missing")

        self.train_src_folder_path = train_src_folder_path
        self.test_src_folder_path = test_src_folder_path
        self.heightmap_src_folder_path = heightmap_src_folder_path
        self.checkpoint_folder_path = checkpoint_folder_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_epochs = num_epochs

        # stipulate the training device
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # choose acceleration platform
        if torch.cuda.is_available():
            if device is not None:
                self.device = device
            else:
                self.device = 'cuda'
        else:
            print("\n[INFO] current device does not support CUDA computation, switch to CPU calculation")
            self.device = 'cpu'

        self.img_transforms = {
            'train': transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
        }

        self.trainset = ImageDataset.Image2HeightmapDataset(
            image_src_folder_path=self.train_src_folder_path,
            heightmap_src_folder_path=self.heightmap_src_folder_path,
            transform=self.img_transforms['train']
        )
        self.testset = ImageDataset.Image2HeightmapDataset(
            image_src_folder_path = self.test_src_folder_path,
            heightmap_src_folder_path=self.heightmap_src_folder_path,
            transform=self.img_transforms['test']
        )

        self.model = UNet.UNet_modified().to(self.device)
        self.loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, verbose=True)

        self.test_loss_min = np.Inf

    def train(self, cur_epoch):
        # the train loader for the whole dataset
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

        self.model.train(True)
        print("Training...")

        # to record the sum of training loss in all batches
        train_loss_sum = 0

        # start to input images and corresponding normalized matrix
        loop = tqdm(enumerate(train_loader, 0), total=len(train_loader), position=0)
        for batch_idx, data in loop:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            labels = labels.to(torch.float32)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward propagation
            outputs = self.model(inputs)
            outputs = outputs.to(torch.float32)
            # calculate loss
            loss = self.loss_fn(outputs, labels)
            # add loss to the sum
            train_loss_sum += loss
            # loss back propagation
            loss.backward()
            # update the parameter
            self.optimizer.step()
            torch.cuda.empty_cache()
            loop.set_description(f'Epoch [{cur_epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=loss.item())

        print("\n[INFO] Epoch {0}/{1} average training loss: {2}".format(cur_epoch, self.num_epochs, train_loss_sum / len(train_loader.dataset) * 1.0))
        self.scheduler.step(train_loss_sum / len(train_loader.dataset) * 1.0)

    def test(self, cur_epoch):
        # the test loader for the whole dataset
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=False, num_workers=4)

        # State that you are testing the model; this prevents layers e.g. Dropout to take effect
        self.model.eval()
        print("Testing...")

        # to record the sum of testing loss in all batches
        test_loss_sum = 0

        with torch.no_grad():
            # Iterate over data
            for data in tqdm(test_loader, total=len(test_loader), position=0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                images = images.to(torch.float32)
                labels = labels.to(torch.float32)
                outputs = self.model(images)
                outputs = outputs.to(torch.float32)
                # Calculate & accumulate loss
                test_loss_sum += self.loss_fn(outputs.data, labels).item()

        self.check_optim(test_loss_sum/len(test_loader.dataset))

        print("\n[INFO] Epoch {}/{} average testing loss: {:.4f})\n".format(cur_epoch,
                                                                 self.num_epochs,
                                                                 test_loss_sum/len(test_loader.dataset)))
        print("--------------------------------------------------------------")

    def save_model(self, timestamp):
        save_path = os.path.join(self.checkpoint_folder_path,
                            'PredictorV2_image2heightmap_{}'
                            .format(time.strftime('%Y-%m-%d_%H-%M-%S', timestamp)))

        if not os.path.exists(save_path):
            print("\n[INFO] Create folder for storing current Checkpoint: {}".format(save_path))
            os.makedirs(save_path)

        torch.save(self.model, os.path.join(save_path,
                                            'model.pth'))

    def check_optim(self, test_loss):
        if test_loss < self.test_loss_min:
            print('\n[INFO] average testing loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(self.test_loss_min, test_loss))
            self.save_model(self.time_start)
            self.test_loss_min = test_loss

    def start(self):
        self.time_start = time.localtime()
        # log the file into result
        sys.stdout = Logger.Logger("result_{}.txt".format(time.strftime('%Y-%m-%d_%H-%M-%S', self.time_start)))
        for epoch in range(1, self.num_epochs+1):
            self.train(epoch)
            self.test(epoch)

class EnergyPredictorV2_heightmap2powerclass():
    def __init__(self,
                 train_src_folder_path: str,
                 test_src_folder_path: str,
                 heightmap_src_folder_path: str,
                 classification_datafile_src_path: str,
                 checkpoint_folder_path: str,
                 train_batch_size: int = 16,
                 test_batch_size: int = 16,
                 num_epochs: int = 5,
                 num_classes: int = 5,
                 device: str = None):

        if train_src_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'train_src_folder_path' for "
                                 "EnergyPredictorV2_heightmap2powerclass is missing")
        if test_src_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'test_src_folder_path' for "
                                 "EnergyPredictorV2_heightmap2powerclass is missing")
        if heightmap_src_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'heightmap_src_folder_path' for "
                                 "EnergyPredictorV2_heightmap2powerclass is missing")
        if classification_datafile_src_path is None:
            raise AttributeError("[ERROR] the attribute 'classification_datafile_src_path' for "
                                 "EnergyPredictorV2_heightmap2powerclass is missing")
        if checkpoint_folder_path is None:
            raise AttributeError("[ERROR] the attribute 'checkpoint_folder_path' for "
                                 "EnergyPredictorV2_heightmap2powerclass is missing")

        self.train_src_folder_path = train_src_folder_path
        self.test_src_folder_path = test_src_folder_path
        self.heightmap_src_folder_path = heightmap_src_folder_path
        self.classification_datafile_src_path = classification_datafile_src_path
        self.checkpoint_folder_path = checkpoint_folder_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs

        # stipulate the training device
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # choose acceleration platform
        if torch.cuda.is_available():
            if device is not None:
                self.device = device
            else:
                self.device = 'cuda'
        else:
            print("\n[INFO] current device does not support CUDA computation, switch to CPU calculation")
            self.device = 'cpu'

        self.heightmap_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }

        self.trainset = ImageDataset.Heightmap2PowerclassDataset(
            img_src_folder_path=self.train_src_folder_path,
            heightmap_src_folder_path=self.heightmap_src_folder_path,
            classification_datafile_src_path=self.classification_datafile_src_path,
            transform=self.heightmap_transforms['train']
        )
        self.testset = ImageDataset.Heightmap2PowerclassDataset(
            img_src_folder_path=self.test_src_folder_path,
            heightmap_src_folder_path=self.heightmap_src_folder_path,
            classification_datafile_src_path=self.classification_datafile_src_path,
            transform=self.heightmap_transforms['test']
        )

        self.model = ResNet.resnet34(self.num_classes).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5,
                                                                    verbose=True)

        self.test_loss_min = np.Inf

    def train(self, cur_epoch):
        # the train loader for the whole dataset
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=True,
                                                   num_workers=4)
        self.model.train(True)
        print("Training...")

        # to record the sum of training loss in all batches
        train_loss_sum = 0
        # to record the sum of correct classification
        num_correct = 0

        # start to input images and corresponding normalized matrix
        loop = tqdm(enumerate(train_loader, 0), total=len(train_loader), position=0)
        for batch_idx, data in loop:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs = inputs.to(torch.float32)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward propagation
            outputs = self.model(inputs)
            # calculate loss
            loss = self.loss_fn(outputs, labels)
            # add loss to the sum
            train_loss_sum += loss
            # loss back propagation
            loss.backward()
            # predict according to probability
            _, pred = torch.max(outputs, 1)
            num_correct += (pred == labels).sum().item()
            running_train_acc = float((pred == labels).sum().item()) / float(len(inputs))
            # update the parameter
            self.optimizer.step()
            loop.set_description(f'Epoch [{cur_epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=loss.item(), acc=running_train_acc)

        print("\n[INFO] Epoch {}/{} average training loss: {}, accuracy: {}/{} ({:.0f}%)\n".format(cur_epoch,
                                                                                                self.num_epochs,
                                                                                                train_loss_sum / len(train_loader.dataset) * 1.0,
                                                                                                num_correct,
                                                                                                len(train_loader.dataset),
                                                                                                num_correct / len(train_loader.dataset) * 100.0))
        self.scheduler.step(train_loss_sum / len(train_loader.dataset) * 1.0)

    def test(self, cur_epoch):
        # the test loader for the whole dataset
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=False, num_workers=4)

        # State that you are testing the model; this prevents layers e.g. Dropout to take effect
        self.model.eval()
        print("Testing...")

        # to record the sum of testing loss in all batches
        test_loss_sum = 0
        # to record the sum of correct classification
        num_correct = 0

        with torch.no_grad():
            # Iterate over data
            for data in tqdm(test_loader, total=len(test_loader), position=0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                images = images.to(torch.float32)
                outputs = self.model(images)
                # Calculate & accumulate loss
                test_loss_sum += self.loss_fn(outputs.data, labels).item()
                # predict according to probability
                _, pred = torch.max(outputs, 1)
                num_correct += (pred == labels).sum().item()

        self.check_optim(test_loss_sum / len(test_loader.dataset))

        print("\n[INFO] Epoch {}/{} average testing loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n".format(cur_epoch,
                                                                                                       self.num_epochs,
                                                                                                       test_loss_sum/len(test_loader.dataset) * 1.0,
                                                                                                       num_correct,
                                                                                                       len(test_loader.dataset),
                                                                                                       num_correct/len(test_loader.dataset) * 100.0))
        print("--------------------------------------------------------------")

    def save_model(self, timestamp):
        save_path = os.path.join(self.checkpoint_folder_path,
                            'PredictorV2_heightmap2powerclass_{}'
                            .format(time.strftime('%Y-%m-%d_%H-%M-%S', timestamp)))

        if not os.path.exists(save_path):
            print("\n[INFO] Create folder for storing current Checkpoint: {}".format(save_path))
            os.makedirs(save_path)

        torch.save(self.model, os.path.join(save_path,
                                            'model.pth'))

    def check_optim(self, test_loss):
        if test_loss < self.test_loss_min:
            print('\n[INFO] average testing loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(self.test_loss_min, test_loss))
            self.save_model(self.time_start)
            self.test_loss_min = test_loss

    def start(self):
        self.time_start = time.localtime()
        # log the file into result
        sys.stdout = Logger.Logger("result_{}.txt".format(time.strftime('%Y-%m-%d_%H-%M-%S', self.time_start)))
        for epoch in range(1, self.num_epochs+1):
            self.train(epoch)
            self.test(epoch)