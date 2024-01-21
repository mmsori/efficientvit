from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam_export import ExportEfficientSam
import torch
import argparse
import warnings


def export_coremltools(args):
    import coremltools as ct

    efficientvit_sam = create_sam_model(
        args.model, True, args.checkpoint, img_size=1024).eval()
    sam = ExportEfficientSam(efficientvit_sam, "coremltools", True)
    sam.eval()

    # Trace the model with random data.
    dummy_inputs = {
        "image": torch.randn(1, 1024, 1024, 3, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 3, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 3), dtype=torch.float),
        # "mask_input": torch.randn(1, 1, 256, 256, dtype=torch.float),
        # "has_mask_input": torch.tensor([1], dtype=torch.float),
        "org_img_shape": torch.tensor([443, 553], dtype=torch.int64),
    }
    traced_model = torch.jit.trace(sam, list(dummy_inputs.values()))
    image_shape = ct.Shape(
        shape=(
            1,
            ct.RangeDim(lower_bound=128, upper_bound=2048, default=1024),
            ct.RangeDim(lower_bound=128, upper_bound=2048, default=1024),
            3,
        )
    )
    point_coord_shape = ct.Shape(shape=(1, ct.RangeDim(
        lower_bound=1, upper_bound=10, default=2), 2))
    point_label_shape = ct.Shape(
        shape=(1, ct.RangeDim(lower_bound=1, upper_bound=10, default=2)))
    # mask_input_shape = ct.Shape((1, 1, 256, 256))
    # has_mask_input_shape = ct.Shape((1,))
    org_img_shape_shape = ct.Shape((2,), default=[512, 512])

    # Convert the model with input_shape.
    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=image_shape, name="image"),
            ct.TensorType(shape=point_coord_shape, name="point_coords"),
            ct.TensorType(shape=point_label_shape, name="point_labels"),
            # ct.TensorType(shape=mask_input_shape, name="mask_input"),
            # ct.TensorType(shape=has_mask_input_shape, name="has_mask_input"),
            ct.TensorType(shape=org_img_shape_shape, name="org_img_shape"),
        ],
        outputs=[ct.TensorType(name="masks"), ct.TensorType(
            name="iou_predictions")],
        convert_to="mlprogram",
    )

    model.save("sam_encoder.mlpackage")


def export_onnx(args):
    print("Loading model...")
    efficientvit_sam = create_sam_model(
        args.model, True, args.checkpoint, img_size=1024).eval()
    # sam = EfficientViTSamPredictor(efficientvit_sam)
    onnx_model = ExportEfficientSam(
        efficientvit_sam, "onnx", include_batch_axis=True, image_resize=args.include_resize)
    onnx_model.eval()
    if args.include_resize:
        dynamic_axes = {
            "image": {1: "width", 2: "height"},
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
    else:
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }

    # embed_dim = sam.prompt_encoder.embed_dim
    # embed_size = sam.prompt_encoder.image_embedding_size
    dummy_inputs = {
        "image": torch.randn(1, 1024, 1024, 3, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 3, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 3), dtype=torch.float),
        # "mask_input": torch.randn(1, 256, 256, dtype=torch.float),
        # "has_mask_input": torch.tensor([0], dtype=torch.float),
        "org_img_shape": torch.tensor([1024, 1024], dtype=torch.int64),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
    with open(args.output, "wb") as f:
        print(f"Exporting onnx model to {args.output}...")
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="l2")
    parser.add_argument("--checkpoint", type=str,
                        default="assets/checkpoints/l2.pt")
    parser.add_argument("--format", type=str, default="onnx")
    parser.add_argument("--output", type=str, default="onnx_output/sam.onnx")
    parser.add_argument("--include-resize", action='store_true')
    args = parser.parse_args()
    if args.format == "onnx":
        export_onnx(args)
    elif args.format == "coreml":
        export_coremltools(args)
