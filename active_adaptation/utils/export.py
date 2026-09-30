import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase


def _patch_onnx_version_converter():
    """Work around an onnxscript 0.3.0 ordering bug.

    `ConvertVersionPass` runs `InlinePass` and then
    `_ConvertVersionPassRequiresInline`, which does:

        if model.functions:  raise ValueError("model contains functions ...")
        if model_opset == target_version:  return   # nothing to do

    The two checks are the wrong way round. torchlib emits opset 18 and
    torch's TORCHLIB_OPSET is also 18, so the conversion is a no-op -- but if
    InlinePass leaves any function behind, the pass raises instead of taking
    the early exit. Reinstate the short-circuit ahead of the functions check;
    when a real conversion IS needed we fall through to the original code.
    """
    try:
        from onnxscript.version_converter import _ConvertVersionPassRequiresInline as P
        from onnxscript import ir
    except Exception:
        return  # different onnxscript version; nothing to patch

    if getattr(P, "_hdmi_opset_shortcircuit", False):
        return

    _orig_call = P.call

    def call(self, model):
        opset = model.graph.opset_imports.get("")
        if opset is not None and opset == self.target_version:
            return ir.passes.PassResult(model, False)
        return _orig_call(self, model)

    P.call = call
    P._hdmi_opset_shortcircuit = True


_patch_onnx_version_converter()


@torch.inference_mode()
def export_onnx(module: ModBase, td: TensorDictBase, path: str, meta=None):
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()
    print(torch.__version__)
    # breakpoint()
    onnx_program = torch.onnx.dynamo_export(module, **td.to_dict())
    onnx_program.save(path)
    print(f"Exported ONNX model to {path}.")

    import json

    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = module.in_keys
    meta["out_keys"] = module.out_keys
    meta["in_shapes"] = ([td[k].shape for k in module.in_keys],)

    json.dump(meta, open(meta_path, "w"), indent=4)
    print(f"Exported metadata to {meta_path}.")

    import onnxruntime as ort

    ort_session = ort.InferenceSession(
        path.replace(".pt", ".onnx"), providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnx_input = tuple(td[k] for k in module.in_keys)
    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    ort_output = ort_session.run(None, onnxruntime_input)
    assert len(ort_output) == len(module.out_keys)
