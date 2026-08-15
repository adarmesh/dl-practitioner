# Fixing the `onnx-tf` ImportError in `C3_M4_Lab_2_onnx.ipynb`

## The problem

The imports cell failed, so nothing after it ran:

```
ImportError: cannot import name 'mapping' from 'onnx'
  onnx_tf/backend.py:25 → onnx_tf/common/data_type.py:4 → from onnx import mapping
```

`onnx-tf` (`onnx-tensorflow`) was archived in 2022. It needs `onnx.mapping`, removed in
onnx 1.16 — this env has onnx 1.22. Shimming `onnx.mapping` back in gets past that line and
then dies on the next incompatibility: `onnx_tf/handlers/backend/bernoulli.py` imports
`tensorflow_probability`, which has no build for TF 2.21. Dead end; the library was replaced,
not merely broken.

## The fix

Replace `onnx-tf` with [`onnx2tf`](https://github.com/PINTO0309/onnx2tf), the maintained
ONNX → TensorFlow converter.

### `pyproject.toml`

```diff
+    "ai-edge-litert>=2.2.0",
-    "onnx-tf>=1.10.0",
+    "onnx-graphsurgeon>=0.6.1",
+    "onnx2tf>=1.28.8",
+    "sng4onnx>=2.0.1",
```

`tf-keras` is also required (`onnx2tf/onnx2tf.py` imports it at module level), as is
`ai-edge-litert` (`onnx2tf/utils/common_functions.py:20` →
`from ai_edge_litert.interpreter import Interpreter`).

### Notebook cells

| Cell | Change |
| --- | --- |
| `7170778f` (imports) | `from onnx_tf.backend import prepare` → `import onnx2tf` |
| `66a6544f` | `prepare(onnx_model)` → `onnx.checker.check_model(onnx_model)` — onnx2tf has no in-memory "TF representation" step, so this cell now just validates the exported graph |
| `662ace2b` | `tf_rep.export_graph(path)` → `onnx2tf.convert(input_onnx_file_path=..., output_folder_path=tf_model_path, non_verbose=True)` |
| `321c234f` | Added `input_data.transpose(0, 2, 3, 1)` before building the input tensor |
| `54a16022`, `5fb99f95`, `0fd713d6` | Markdown updated, incl. a note on why `onnx-tf` was dropped |

## The one behavioural difference: tensor layout

`onnx2tf` rewrites the graph into TensorFlow's native **channel-last (NHWC)** layout. The
SavedModel therefore expects `(N, 224, 224, 3)`, not PyTorch's `(N, 3, 224, 224)`:

```python
input_data_nhwc = input_data.transpose(0, 2, 3, 1)
input_tensor = tf.convert_to_tensor(input_data_nhwc, dtype=tf.float32)
```

`show_prediction_grid` still receives the original NCHW `input_data` for plotting.

Everything else in the optional section is unchanged: the `input` / `output` signature names
and the `dynamic_axes` batch dimension both survive conversion.

## Verification

- Imports cell runs clean: onnx 1.22.0, onnxruntime 1.28.0, tensorflow 2.21.0.
- End-to-end check on a standalone resnet18 with a 28-class head, exported at opset 11 with
  the notebook's exact `torch.onnx.export` arguments, then converted with `onnx2tf`:
  SavedModel signature kept `input` / `output`, output shape `(9, 28)`, **max abs diff vs
  PyTorch = 4.8e-6**.

### Not run

The notebook's own cells were not executed — training needs `./fruit_and_vegetable_subset`,
which is not present in this lab directory, so the real `fruit_veg_model.onnx` was never
produced or converted. The graph is architecturally identical to the one tested above, but
the notebook itself remains unverified end to end.
