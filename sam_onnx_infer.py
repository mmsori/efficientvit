import numpy as np
import argparse
import cv2
import onnxruntime


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0])], axis=0)
    else:
        color = np.array([255 / 255, 144 / 255, 40 / 255, 0.6])
    h, w = mask.shape[-2:]
    alpha = mask > 0
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image[:, :, 3] = alpha * 0.6
    mask_image *= 255
    return mask_image


def image_overlay(image, mask):
    overlay_alpha = mask[:, :, 3:] / 255
    overlayed_img = mask[:, :, :3] * \
        overlay_alpha + image * (1 - overlay_alpha)
    return overlayed_img


class InferModule:
    def __init__(self, args):
        self.model = onnxruntime.InferenceSession(args.model)

    def infer(self, image, input_point, input_label):
        input_point = np.array(input_point)
        input_label = np.array(input_label)
        onnx_coord = np.concatenate([input_point, np.array(
            [[0.0, 0.0]])], axis=0).astype(np.float32)[None, ...]
        onnx_label = np.concatenate([input_label, np.array(
            [-1])], axis=0).astype(np.float32)[None, ...]

        # onnx_mask_input = np.zeros((1, 256, 256), dtype=np.float32)
        # onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        from PIL import Image

        example_image = Image.open(image)
        example_image = np.array(example_image).astype(
            np.float32)[None, :, :, :3]
        _, h, w, _ = example_image.shape
        original_size = np.array([h, w], dtype=np.int64)
        inputs = {
            "image": example_image,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "org_img_shape": original_size,
        }
        out_dict = self.model.run(None, inputs)
        mask = out_dict[0]
        mask = mask.squeeze()

        # mask = show_mask(mask)
        # overlay_img = image_overlay(example_image, mask).squeeze()
        # cv2.imwrite("img.jpg", cv2.cvtColor(
        #     overlay_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        cv2.imwrite(args.output, cv2.cvtColor(
            mask.astype(np.uint8), cv2.COLOR_BGR2RGB))
        return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="sample/dolphin.jpg")
    parser.add_argument("--output", type=str, default="sample/mask.jpg")
    parser.add_argument("--model", type=str, default="onnx_output/sam.onnx")
    args = parser.parse_args()
    module = InferModule(args)
    input_point = [[199, 157]]
    input_label = [1]
    module.infer(args.input, input_point, input_label)
