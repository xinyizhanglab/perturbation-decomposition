"""Check the installed CUDA kernels before loading a native checkpoint."""

import importlib
from importlib.metadata import version
import json


def check_gpu(device="cuda:0", extensions=()):
    from benchmarks._runtime import require_compute
    require_compute("CUDA runtime check")
    if not device.startswith("cuda"):
        raise ValueError("GPU checks require a CUDA device.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")
    report = {
        "device": device,
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "compiled_architectures": torch.cuda.get_arch_list(),
        "extensions": {},
    }
    try:
        with torch.inference_mode():
            identity = torch.eye(2, device=device)
            if (identity @ identity).sum().item() != 2:
                raise RuntimeError("CUDA matrix multiplication returned an unexpected value.")
            for name in extensions:
                module = importlib.import_module(name)
                if name == "torch_scatter":
                    result = module.scatter(
                        torch.ones(2, device=device),
                        torch.zeros(2, dtype=torch.long, device=device), dim=0)
                elif name == "torch_sparse":
                    indices = torch.zeros(1, dtype=torch.long, device=device)
                    matrix = module.SparseTensor(
                        row=indices, col=indices, value=torch.ones(1, device=device),
                        sparse_sizes=(1, 1))
                    result = matrix.matmul(torch.ones((1, 1), device=device))
                else:
                    raise ValueError(f"No CUDA probe is defined for {name}.")
                if not torch.isfinite(result).all().item():
                    raise RuntimeError(f"{name} returned nonfinite probe values.")
                torch.cuda.synchronize(device)
                report["extensions"][name] = version(name.replace("_", "-"))
            torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"CUDA runtime probe failed: {exc}\n"
            f"Runtime: {json.dumps(report)}\n"
            "Use the model's locked uv environment with CUDA wheels supporting "
            "this GPU and matching PyG extensions; no CPU fallback was applied."
        ) from exc
    report["kernel_checks_passed"] = True
    return report
