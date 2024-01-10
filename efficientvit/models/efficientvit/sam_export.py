from .sam import EfficientViTSamPredictor, EfficientViTSam
from typing import Literal
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..utils import ResizeLongestSide
import numpy as np


class PreProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = ResizeLongestSide(1024)
        self.pixel_mean = torch.tensor(
            [123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.img_size = torch.tensor(1024, dtype=torch.int64)

    def preprocess(self, x: torch.Tensor, transformed_size: torch.Tensor) -> np.ndarray:
        # Normalize colors

        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        # padh = img_size - transformed_size[0]
        # padw = img_size - transformed_size[1]
        new_x = torch.zeros(1, 3, self.img_size, self.img_size)
        new_x[..., : transformed_size[0], : transformed_size[1]] = x
        # x = F.pad(x, (0, padw, 0, padh))
        return new_x

    def forward(self, image: torch.Tensor, point_coords: torch.Tensor, org_img_shape: torch.Tensor):
        image = image.permute(0, 3, 1, 2)
        coord = self.transform.apply_coords_torch(point_coords, org_img_shape)
        image, target_size = self.transform.apply_image_torch(
            image, org_img_shape)
        image = self.preprocess(image, target_size)
        return image, coord, target_size


class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask: torch.Tensor, target_size, org_img_shape, ch=3):
        masks: torch.Tensor = F.interpolate(
            mask,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : target_size[0], : target_size[1]]
        masks = F.interpolate(masks, size=(
            org_img_shape[0], org_img_shape[1]), mode="bilinear", align_corners=False)
        masks = (masks > 0).type(torch.int64) * \
            torch.tensor(255, dtype=torch.int64)
        masks = masks.expand(1, ch, -1, -1)
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.type(torch.uint8)
        return masks


class ImageEncoder(nn.Module):
    def __init__(self, model: EfficientViTSam):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.image_encoder(image)


class SAM(EfficientViTSamPredictor):
    def __init__(self, model: EfficientViTSam):
        super().__init__(model)
        self.constant_mask_input = torch.randn(
            1, 1, 256, 256, dtype=torch.float)
        self.constant_has_mask = torch.tensor([0], dtype=torch.float)

    def forward(self,
                image_embeddings: torch.Tensor,
                point_coords: torch.Tensor,
                point_labels: torch.Tensor):

        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(
            self.constant_mask_input, self.constant_has_mask)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        masks, scores = self.select_masks(masks, scores, point_coords.shape[1])
        return masks, scores


class ExportEfficientSam(nn.Module):
    def __init__(self, sam: EfficientViTSam, format: Literal["onnx", "coremltools"], include_batch_axis=False):
        super().__init__()
        self.preprocess_module = PreProcess()
        self.image_encoder = ImageEncoder(sam)
        self.sam_model = SAM(sam)
        self.postprocess_module = PostProcess()
        self.format = format
        self.sam = sam
        self.include_batch_axis = include_batch_axis

    def forward(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        org_img_shape: torch.Tensor,
    ):

        if not self.include_batch_axis:
            image = image[None, ...]
            point_coords = point_coords[None, ...]
            point_labels = point_labels[None, ...]
        image, point_coords, target_size = self.preprocess_module(
            image, point_coords, org_img_shape)
        image_embedding = self.image_encoder(image)
        mask, score = self.sam_model(
            image_embedding, point_coords, point_labels)
        if self.format == "coremltools":
            return mask, score
        mask_4ch = self.postprocess_module(mask, target_size, org_img_shape)

        return mask_4ch, score
