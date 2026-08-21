# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:39:01 2025

@author: ae1028
"""

import os
print(os.getcwd())

import sys
sys.path.append(r"F:\3Spring 2025\Woodchip\Report26_Synthetic Data Generation_cutout")

import numpy as np
import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection

# Define DETR Model with PyTorch Lightning
class DetrModel(pl.LightningModule):
    def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
        super().__init__()

        # Define DETR model
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101",
            num_labels=1,  
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # ✅ Loss storage (for smooth plotting)
        self.training_losses = []
        self.validation_losses = []
        self.current_training_losses = []  # Stores per batch, reset every epoch
        self.current_validation_losses = []

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        
        # Convert labels to correct device
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        # Forward pass
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
    
        # ✅ Store per-batch loss for averaging
        self.current_training_losses.append(loss.item())
    
        # ✅ Explicitly set batch size to avoid warning
        batch_size = batch["pixel_values"].size(0)
        
        # ✅ Log training loss
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), prog_bar=False, logger=True, batch_size=batch_size)
    
        return loss


    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
    
        # ✅ Store per-batch validation loss for averaging
        self.current_validation_losses.append(loss.item())
    
        # ✅ Explicitly set batch size
        batch_size = batch["pixel_values"].size(0)
        
        # ✅ Log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v.item(), prog_bar=False, logger=True, batch_size=batch_size)
    
        return loss


    def on_train_epoch_end(self):
        """Store the averaged loss for the entire epoch"""
        if self.current_training_losses:
            avg_loss = np.mean(self.current_training_losses)
            self.training_losses.append(avg_loss)
            self.current_training_losses = []  # Reset for next epoch

    def on_validation_epoch_end(self):
        """Store the averaged validation loss for the entire epoch"""
        if self.current_validation_losses:
            avg_loss = np.mean(self.current_validation_losses)
            self.validation_losses.append(avg_loss)
            self.current_validation_losses = []  # Reset for next epoch

    def configure_optimizers(self):
        """Optimizer with different learning rate for the backbone."""
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer