# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from segment_anything.utils.amg import build_all_layer_point_grids

# from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize, to_pil_image

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
    build_norm,
)
from efficientvit.models.utils import get_device, ResizeLongestSide

__all__ = [
    "SamPad",
    "SamResize",
    "SamNeck",
    "EfficientViTSamImageEncoder",
    "EfficientViTSam",
    "EfficientViTSamPredictor",
    "EfficientViTSamAutomaticMaskGenerator",
    "efficientvit_sam_l0",
    "efficientvit_sam_l1",
    "efficientvit_sam_l2",
]


class SamPad:
    def __init__(self, size: int, fill: float = 0, pad_mode="corner") -> None:
        self.size = size
        self.fill = fill
        self.pad_mode = pad_mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        th, tw = self.size, self.size
        assert th >= h and tw >= w
        if self.pad_mode == "corner":
            image = F.pad(image, (0, tw - w, 0, th - h), value=self.fill)
        else:
            raise NotImplementedError
        return image

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size},mode={self.pad_mode},fill={self.fill})"


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class SamNeck(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm="bn2d",
        act_func="gelu",
    ):
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "sam_encoder": OpSequential(
                [
                    ConvLayer(
                        head_width,
                        out_dim,
                        1,
                        use_bias=True,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SamNeck, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSamImageEncoder(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone or EfficientViTLargeBackbone, neck: SamNeck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

        self.norm = build_norm("ln2d", 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.neck(feed_dict)

        output = feed_dict["sam_encoder"]
        output = self.norm(output)
        return output


class EfficientViTSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: EfficientViTSamImageEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        image_size: tuple[int, int] = (1024, 512),
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                SamResize(self.image_size[1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
                SamPad(self.image_size[1]),
            ]
        )

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: torch.Tensor,
        original_size: torch.Tensor,
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, size=(original_size[0], original_size[1]), mode="bilinear", align_corners=False)
        return masks


class EfficientViTSamPredictor(nn.Module):
    def __init__(self, sam_model: EfficientViTSam) -> None:
        super().__init__()
        self.model = sam_model
        self.reset_image()

    @property
    def transform(self):
        return self

    @property
    def device(self):
        return get_device(self.model)

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / 1024
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[i].weight * (
                point_labels == i
            )

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (1 - has_mask_input) * self.model.prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        )
        return mask_embedding

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None

    def apply_coords(self, coords: np.ndarray, im_size=None) -> np.ndarray:
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, im_size=None) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    def select_masks(self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int):
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor([[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]).to(
            iou_preds.device
        )
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.inference_mode()
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        self.reset_image()

        self.original_size = image.shape[:2]
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )

        torch_data = self.model.transform(image).unsqueeze(dim=0).to(get_device(self.model))
        self.features = self.model.image_encoder(torch_data)
        self.is_image_set = True

    def predict(
        self,
        point_coords: np.ndarray or None = None,
        point_labels: np.ndarray or None = None,
        box: np.ndarray or None = None,
        mask_input: np.ndarray or None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        device = get_device(self.model)
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = self.apply_coords(point_coords)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.apply_boxes(box)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    @staticmethod
    def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    @torch.inference_mode()
    def predict_torch(
        self,
        point_coords: torch.Tensor or None = None,
        point_labels: torch.Tensor or None = None,
        boxes: torch.Tensor or None = None,
        mask_input: torch.Tensor or None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        #  if not self.is_image_set:
        #    raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def forward(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ):
        image_embeddings = self.model.image_encoder(image)

        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        masks, scores = self.select_masks(masks, scores, point_coords.shape[1])
        # prepadded_size = self.resize_longest_image_size(org_input_size, self.model.image_size[0]).to(torch.int64)
        # #upscaled_masks = self.model.postprocess_masks(masks, prepadded_size, org_input_size)
        # upscaled_masks=masks

        return masks, scores


class EfficientViTSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        model: EfficientViTSam,
        points_per_side: int or None = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: list[np.ndarray] or None = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = EfficientViTSamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode


def build_efficientvit_sam(image_encoder: EfficientViTSamImageEncoder, image_size: int) -> EfficientViTSam:
    return EfficientViTSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        image_size=(1024, image_size),
    )


def efficientvit_sam_l0(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l0

    backbone = efficientvit_backbone_l0(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l1(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l2(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)
