# End-to-end SAM ONNX export for EfficientViT

Include input preprocess(resizing, NHWC to NCHW) and output mask postprocess in a single model.

The input image,and point prompt input should include batch axis.  

The output mask has same shape as an original input image.

- Export usage
```bash
python sam_export.py --model {model_type} --checkpoint {path_to_model_weight} --output {output_path} --include-resize
```
If you don't want to incldue pre/postprocessing step in the ONNX model, just omit `--include-resize`.

## TODO
- code refactoring
- implement auto mask generation
