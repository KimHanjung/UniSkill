#!/usr/bin/env python3
"""Extract latent actions with a pretrained IDM from LIBERO demonstrations."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp
import time
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from dynamics.idm import IDM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory that contains LIBERO *.pkl demonstrations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to store latent actions (mirrors data structure).",
    )
    parser.add_argument(
        "--idm-checkpoint",
        type=Path,
        required=True,
        help="Path to the pretrained IDM checkpoint (.pth).",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face identifier or local path for the depth estimator.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="pixels",
        choices=["pixels", "pixels_egocentric"],
        help="Which camera view to use from the stored observations.",
    )
    parser.add_argument(
        "--skill-interval",
        type=int,
        default=1,
        help="Temporal offset between frames when forming IDM pairs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of frame pairs to process per forward pass.",
    )
    parser.add_argument(
        "--idm-resolution",
        type=int,
        default=224,
        help="Resolution expected by the IDM (both width and height).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of transformer layers for IDM instantiation.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads for IDM instantiation.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Transformer hidden dimension for IDM instantiation.",
    )
    parser.add_argument(
        "--skill-dim",
        type=int,
        default=64,
        help="Latent skill dimension produced by the IDM.",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=768,
        help="Output dimension used during training (unused when return_skill=True).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute latents even if the output file already exists.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while processing demonstrations.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        help="CUDA device ids for multi-GPU inference (e.g. --devices 0 1).",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=2,
        help="Number of background threads to prepare batches; 0 disables prefetch.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable torch.cuda.amp autocast during model forward passes.",
    )
    return parser.parse_args()


def resolve_devices(args: argparse.Namespace) -> List[torch.device]:
    if args.devices:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when specifying --devices.")
        device_ids = list(dict.fromkeys(args.devices))
        if not device_ids:
            raise ValueError("--devices must include at least one CUDA id.")
        devices = [torch.device(f"cuda:{idx}") for idx in device_ids]
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but no GPUs are available.")
            if device.index is None:
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
        devices = [device]
    return devices


def load_idm(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    idm = IDM(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        skill_dim=args.skill_dim,
        out_dim=args.out_dim,
        idm_resolution=args.idm_resolution,
    )
    checkpoint = torch.load(args.idm_checkpoint, map_location="cpu")
    idm.load_state_dict(checkpoint)
    idm.requires_grad_(False)
    idm.eval()
    idm.to(device)
    return idm, idm.skill_proj.out_features


def load_depth_estimator(
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(model_name)
    depth_estimator.requires_grad_(False)
    depth_estimator.eval()
    depth_estimator.to(device)
    return depth_estimator


def get_predicted_depth(output):
    if hasattr(output, "predicted_depth"):
        return output.predicted_depth
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unexpected depth estimator output type: {type(output)}")



def distribute_files(files: List[Path], num_groups: int) -> List[List[Path]]:
    groups = [[] for _ in range(num_groups)]
    for idx, file_path in enumerate(files):
        groups[idx % num_groups].append(file_path)
    return groups





def iter_demo_files(data_root: Path) -> Iterable[Path]:
    for file_path in sorted(data_root.rglob("*.pkl")):
        if file_path.is_file():
            yield file_path


def to_numpy_frames(sequence: Sequence, camera_key: str) -> np.ndarray:
    if isinstance(sequence, dict):
        frames = sequence[camera_key]
    else:
        raise ValueError("Expected per-demo observation dict.")
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frame shape {frames.shape}")
    return frames


def extract_latents_for_demo(
    frames: np.ndarray,
    *,
    idm: torch.nn.Module,
    idm_resolution: int,
    depth_processor: AutoImageProcessor,
    depth_estimator: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    interval: int,
    skill_dim: int,
    prefetch_workers: int,
    use_amp: bool,
) -> np.ndarray:
    num_pairs = frames.shape[0] - interval
    if num_pairs <= 0:
        return np.empty((0, skill_dim), dtype=np.float32)

    latents: List[torch.Tensor] = []
    pin_memory = device.type == "cuda"
    autocast_enabled = use_amp and device.type == "cuda"

    def prepare_batch(start: int, end: int):
        curr_indices = np.arange(start, end)
        next_indices = curr_indices + interval

        curr_tensor = (
            torch.from_numpy(frames[curr_indices])
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        next_tensor = (
            torch.from_numpy(frames[next_indices])
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        if pin_memory:
            curr_tensor = curr_tensor.pin_memory()
            next_tensor = next_tensor.pin_memory()

        depth_inputs = [frames[idx] for idx in curr_indices] + [frames[idx] for idx in next_indices]
        depth_batch = depth_processor(
            images=depth_inputs,
            do_rescale=False,
            return_tensors="pt",
        )
        if pin_memory:
            depth_batch = {k: v.pin_memory() for k, v in depth_batch.items()}

        return curr_tensor, next_tensor, depth_batch

    executor = ThreadPoolExecutor(max_workers=prefetch_workers) if prefetch_workers > 0 else None

    def submit(start: int):
        end = min(start + batch_size, num_pairs)
        if executor is None:
            return prepare_batch(start, end)
        return executor.submit(prepare_batch, start, end)

    try:
        future = submit(0) if executor is not None else None
        start = 0
        while start < num_pairs:
            end = min(start + batch_size, num_pairs)
            if executor is not None:
                next_future = submit(end) if end < num_pairs else None
                curr_tensor, next_tensor, depth_batch = future.result()
            else:
                curr_tensor, next_tensor, depth_batch = submit(start)
                next_future = None

            visual_curr = F.interpolate(
                curr_tensor.to(device, non_blocking=True),
                size=(idm_resolution, idm_resolution),
                mode="bilinear",
                align_corners=False,
            )
            visual_next = F.interpolate(
                next_tensor.to(device, non_blocking=True),
                size=(idm_resolution, idm_resolution),
                mode="bilinear",
                align_corners=False,
            )
            visual_pair = torch.stack([visual_curr, visual_next], dim=1)

            depth_batch = {k: v.to(device, non_blocking=True) for k, v in depth_batch.items()}
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                depth_outputs = depth_estimator(**depth_batch)
                depth_outputs = get_predicted_depth(depth_outputs)
                if depth_outputs.ndim == 4 and depth_outputs.size(1) == 1:
                    depth_outputs = depth_outputs.squeeze(1)

                curr_depth, next_depth = torch.chunk(depth_outputs, 2, dim=0)
                depth_pair = torch.stack([curr_depth, next_depth], dim=1)
                depth_pair = F.interpolate(
                    depth_pair,
                    size=(idm_resolution, idm_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                skills = idm(depth_pair, visual_pair, return_skill=True)

            latents.append(skills.float().cpu())

            if executor is not None:
                future = next_future
            start = end
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    stacked = torch.cat(latents, dim=0)
    return stacked.numpy().astype(np.float32, copy=False)

def should_skip_scene(file_path: Path, args: argparse.Namespace) -> bool:
    relative = file_path.relative_to(args.data_root)
    scene_out = (args.output_root / relative).with_suffix(".npy")
    return scene_out.exists() and not args.overwrite

def process_demonstration_file(
    file_path: Path,
    *,
    args: argparse.Namespace,
    idm: torch.nn.Module,
    depth_processor: AutoImageProcessor,
    depth_estimator: torch.nn.Module,
    device: torch.device,
    skill_dim: int,
    prefetch_workers: int,
    use_amp: bool,
) -> List[Path]:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    observations = data["observations"]
    if not isinstance(observations, (list, tuple)):
        raise ValueError(f"Unexpected observations container in {file_path}")

    relative = file_path.relative_to(args.data_root)
    scene_out = (args.output_root / relative).with_suffix(".npy")
    scene_out.parent.mkdir(parents=True, exist_ok=True)
    if scene_out.exists() and not args.overwrite:
        return [scene_out]

    scene_latents: List[np.ndarray] = []
    for episode in observations:
        frames = to_numpy_frames(episode, args.camera_key)
        latent_actions = extract_latents_for_demo(
            frames,
            idm=idm,
            idm_resolution=args.idm_resolution,
            depth_processor=depth_processor,
            depth_estimator=depth_estimator,
            device=device,
            batch_size=args.batch_size,
            interval=args.skill_interval,
            skill_dim=skill_dim,
            prefetch_workers=prefetch_workers,
            use_amp=use_amp,
        )
        scene_latents.append(latent_actions)

    packed = np.array(scene_latents, dtype=object)
    np.save(scene_out, packed, allow_pickle=True)

    return [scene_out]

def run_extraction_on_device(
    device: torch.device,
    files: List[Path],
    args: argparse.Namespace,
    show_progress: bool,
    progress_counter: Optional[mp.Value] = None,
) -> None:
    if not files:
        return

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    idm, skill_dim = load_idm(args, device)
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
    depth_estimator = load_depth_estimator(args.depth_model, device)

    iterator: Iterable[Path]
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(files, desc=f"Device {device}: extracting", position=0 if device.index is None else device.index)
    else:
        iterator = files

    for file_path in iterator:
        path_obj = Path(file_path)
        if should_skip_scene(path_obj, args):
            if progress_counter is not None:
                with progress_counter.get_lock():
                    progress_counter.value += 1
            continue
        process_demonstration_file(
            path_obj,
            args=args,
            idm=idm,
            depth_processor=depth_processor,
            depth_estimator=depth_estimator,
            device=device,
            skill_dim=skill_dim,
            prefetch_workers=args.prefetch_workers,
            use_amp=args.amp,
        )
        if progress_counter is not None:
            with progress_counter.get_lock():
                progress_counter.value += 1


def worker_entry(
    device_str: str,
    file_paths: List[str],
    args: argparse.Namespace,
    progress_counter: Optional[mp.Value],
) -> None:
    device = torch.device(device_str)
    files = [Path(p) for p in file_paths]
    run_extraction_on_device(
        device,
        files,
        args,
        show_progress=False,
        progress_counter=progress_counter,
    )




def main() -> None:
    args = parse_args()
    devices = resolve_devices(args)

    args.output_root.mkdir(parents=True, exist_ok=True)

    files = list(iter_demo_files(args.data_root))
    if not files:
        return

    if len(devices) == 1:
        run_extraction_on_device(devices[0], files, args, show_progress=args.progress)
    else:
        groups = distribute_files(files, len(devices))
        ctx = mp.get_context("spawn")
        processes: List[mp.Process] = []
        progress_counter = ctx.Value('i', 0) if args.progress else None
        progress_bar = None
        try:
            if args.progress:
                from tqdm import tqdm

                progress_bar = tqdm(total=len(files), desc="Total progress")
            for device, subset in zip(devices, groups):
                filtered = []
                for path in subset:
                    if should_skip_scene(path, args):
                        if progress_counter is not None:
                            with progress_counter.get_lock():
                                progress_counter.value += 1
                        continue
                    filtered.append(path)
                if not filtered:
                    continue
                proc = ctx.Process(
                    target=worker_entry,
                    args=(str(device), [str(p) for p in filtered], args, progress_counter),
                )
                proc.start()
                processes.append(proc)
            if progress_bar is not None and progress_counter is not None:
                while True:
                    with progress_counter.get_lock():
                        current = progress_counter.value
                    progress_bar.n = current
                    progress_bar.refresh()
                    if current >= len(files) and not any(p.is_alive() for p in processes):
                        break
                    time.sleep(0.2)
                progress_bar.close()
                progress_bar = None
            for proc in processes:
                proc.join()
        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            if progress_bar is not None:
                progress_bar.close()


if __name__ == "__main__":
    main()
