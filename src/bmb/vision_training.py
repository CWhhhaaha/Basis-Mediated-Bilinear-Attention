from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from transformers import ViTConfig, ViTForImageClassification

from bmb.layer_bbt_attention import apply_layer_bbt_attention
from bmb.layer_fully_shared_attention import apply_layer_fully_shared_attention
from bmb.layer_lowrank_attention import apply_layer_lowrank_attention
from bmb.layer_partial_shared_attention import apply_layer_partial_shared_attention
from bmb.layer_partial_qk_shared_attention import apply_layer_partial_qk_shared_attention
from bmb.layer_symmetric_latent_attention import apply_layer_symmetric_latent_attention
from bmb.layer_uv_latent_attention import apply_layer_uv_latent_attention


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def resolve_num_workers(training_config: dict[str, Any]) -> int:
    return int(training_config.get("num_workers", 2))


def regularization_active(epoch: int, schedule: list[dict[str, Any]]) -> bool:
    if not schedule:
        return True
    for item in schedule:
        start = int(item["epoch_start"])
        end = int(item["epoch_end"])
        every = item.get("every_n_epochs")
        if start <= epoch <= end and every:
            return (epoch - start) % int(every) == 0
    return False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


@dataclass
class RunPaths:
    output_dir: Path
    metrics_path: Path
    epoch_metrics_path: Path
    ratio_path: Path
    summary_path: Path
    model_dir: Path


def prepare_run_paths(output_dir: Path) -> RunPaths:
    analysis_dir = output_dir / "analysis"
    model_dir = output_dir / "model"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        output_dir=output_dir,
        metrics_path=output_dir / "metrics.json",
        epoch_metrics_path=analysis_dir / "epoch_metrics.json",
        ratio_path=analysis_dir / "layer_asymmetry_by_epoch.json",
        summary_path=output_dir / "run_summary.json",
        model_dir=model_dir,
    )


def build_vit_model(model_config: dict[str, Any], num_classes: int) -> ViTForImageClassification:
    hidden_size = int(model_config["hidden_size"])
    config = ViTConfig(
        image_size=int(model_config.get("image_size", 32)),
        patch_size=int(model_config.get("patch_size", 4)),
        num_channels=int(model_config.get("num_channels", 3)),
        hidden_size=hidden_size,
        num_hidden_layers=int(model_config["num_hidden_layers"]),
        num_attention_heads=int(model_config["num_attention_heads"]),
        intermediate_size=int(model_config["intermediate_size"]),
        hidden_dropout_prob=float(model_config.get("hidden_dropout_prob", 0.0)),
        attention_probs_dropout_prob=float(model_config.get("attention_dropout_prob", 0.0)),
        num_labels=num_classes,
        qkv_bias=True,
    )
    model = ViTForImageClassification(config)
    variant = str(model_config.get("attention_variant", "standard"))
    if variant == "layer_symmetric_latent":
        model = apply_layer_symmetric_latent_attention(model, model_config)
    elif variant == "layer_uv_latent":
        model = apply_layer_uv_latent_attention(model, model_config)
    elif variant == "layer_bbt":
        model = apply_layer_bbt_attention(model, model_config)
    elif variant == "layer_fully_shared":
        model = apply_layer_fully_shared_attention(model, model_config)
    elif variant == "layer_partial_shared":
        model = apply_layer_partial_shared_attention(model, model_config)
    elif variant == "layer_partial_qk_shared":
        model = apply_layer_partial_qk_shared_attention(model, model_config)
    elif variant == "layer_lowrank":
        model = apply_layer_lowrank_attention(model, model_config)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def visible_cuda_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def maybe_wrap_data_parallel(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type == "cuda" and visible_cuda_device_count() > 1:
        return nn.DataParallel(model)
    return model


def iter_attention_layers(model: nn.Module) -> Iterable[Any]:
    base_model = unwrap_model(model)
    return base_model.vit.encoder.layer


def apply_symmetric_query_key_initialization(model: nn.Module) -> None:
    with torch.no_grad():
        for layer in iter_attention_layers(model):
            attention = layer.attention.attention
            if not hasattr(attention, "query") or not hasattr(attention, "key"):
                continue
            attention.key.weight.copy_(attention.query.weight)
            if attention.query.bias is not None and attention.key.bias is not None:
                attention.key.bias.copy_(attention.query.bias)


def compute_attention_kernel(attention: Any) -> torch.Tensor:
    if hasattr(attention, "effective_layer_kernel"):
        return attention.effective_layer_kernel()
    if hasattr(attention, "query") and hasattr(attention, "key"):
        return attention.query.weight @ attention.key.weight.T
    raise TypeError(f"Unsupported attention module type for kernel extraction: {type(attention)!r}")


def compute_layer_asymmetry_ratios(model: nn.Module) -> list[float]:
    ratios: list[float] = []
    with torch.no_grad():
        for layer in iter_attention_layers(model):
            attention = layer.attention.attention
            gram = compute_attention_kernel(attention).detach()
            asym = 0.5 * (gram - gram.T)
            denom = torch.sum(gram * gram).clamp_min(1e-12)
            ratio = torch.sum(asym * asym) / denom
            ratios.append(float(ratio.item()))
    return ratios


def compute_reg_loss(
    model: nn.Module,
    intervals: list[dict[str, float]],
    penalty: str = "linear_hinge",
) -> tuple[torch.Tensor, list[float]]:
    penalties: list[torch.Tensor] = []
    ratios: list[float] = []
    for layer, interval in zip(iter_attention_layers(model), intervals):
        attention = layer.attention.attention
        gram = compute_attention_kernel(attention)
        asym = 0.5 * (gram - gram.T)
        denom = torch.sum(gram * gram).clamp_min(1e-12)
        ratio = torch.sum(asym * asym) / denom
        lower = torch.relu(torch.tensor(interval["rho_min"], device=ratio.device) - ratio)
        upper = torch.relu(ratio - torch.tensor(interval["rho_max"], device=ratio.device))
        if penalty == "linear_hinge":
            penalties.append(lower + upper)
        elif penalty == "squared_hinge":
            penalties.append(lower.square() + upper.square())
        else:
            raise ValueError(f"Unsupported regularization penalty: {penalty}")
        ratios.append(float(ratio.detach().item()))

    if not penalties:
        zero = next(model.parameters()).new_zeros(())
        return zero, ratios
    return torch.stack(penalties).mean(), ratios


def count_reg_flops(hidden_size: int, num_layers: int) -> int:
    matrix_mul = 2 * (hidden_size**3)
    aux = 8 * (hidden_size**2)
    return num_layers * (matrix_mul + aux)


def count_analysis_flops(hidden_size: int, num_layers: int) -> int:
    return count_reg_flops(hidden_size, num_layers)


def estimate_model_flops(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[int | None, str]:
    try:
        flops = unwrap_model(model).floating_point_ops(batch)
    except Exception:
        return None, "unavailable"
    return int(flops), "transformers.floating_point_ops"


def resolve_global_batch_size(training_config: dict[str, Any], replica_count: int) -> int:
    per_device_batch_size = int(training_config["per_device_batch_size"])
    return per_device_batch_size * max(int(replica_count), 1)


def build_cifar10_loaders(
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    replica_count: int = 1,
) -> tuple[DataLoader, DataLoader]:
    image_size = int(dataset_config.get("image_size", 32))
    mean = tuple(float(x) for x in dataset_config.get("mean", [0.4914, 0.4822, 0.4465]))
    std = tuple(float(x) for x in dataset_config.get("std", [0.2470, 0.2435, 0.2616]))
    data_root = Path(dataset_config["resolved_paths"]["data_root"])
    batch_size = resolve_global_batch_size(training_config, replica_count)
    num_workers = resolve_num_workers(training_config)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def resolve_imagefolder_split_dirs(dataset_config: dict[str, Any]) -> tuple[Path, Path]:
    resolved_paths = dataset_config.get("resolved_paths", {}) if isinstance(dataset_config.get("resolved_paths"), dict) else {}
    train_dir = resolved_paths.get("train_dir")
    val_dir = resolved_paths.get("val_dir")
    data_root = resolved_paths.get("data_root")

    if train_dir and val_dir:
        return Path(train_dir), Path(val_dir)
    if data_root:
        root = Path(data_root)
        candidate_train = root / "train"
        candidate_val = root / "val"
        if candidate_train.exists() and candidate_val.exists():
            return candidate_train, candidate_val
    raise FileNotFoundError(
        "Could not resolve ImageFolder train/val directories. "
        "Expected dataset.train_dir and dataset.val_dir, or dataset.data_root with train/ and val/ subdirectories."
    )


def build_imagenet1k_loaders(
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    replica_count: int = 1,
) -> tuple[DataLoader, DataLoader]:
    image_size = int(dataset_config.get("image_size", 224))
    eval_resize_size = int(dataset_config.get("eval_resize_size", round(image_size / 0.875)))
    mean = tuple(float(x) for x in dataset_config.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in dataset_config.get("std", [0.229, 0.224, 0.225]))
    batch_size = resolve_global_batch_size(training_config, replica_count)
    num_workers = resolve_num_workers(training_config)
    train_dir, val_dir = resolve_imagefolder_split_dirs(dataset_config)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def build_vision_loaders(
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    replica_count: int = 1,
) -> tuple[DataLoader, DataLoader]:
    dataset_name = str(dataset_config.get("name", "cifar10")).lower()
    if dataset_name == "cifar10":
        return build_cifar10_loaders(dataset_config, training_config, replica_count=replica_count)
    if dataset_name in {"imagenet1k", "imagenet-1k", "imagenet_1k", "imagenet"}:
        return build_imagenet1k_loaders(dataset_config, training_config, replica_count=replica_count)
    raise ValueError(f"Unsupported vision dataset: {dataset_name}")


def format_init_summary(
    *,
    experiment_name: str,
    total_epochs: int,
    learning_rate: float,
    analysis_time_sec: float,
    layer_ratios: list[float] | None,
) -> str:
    summary = (
        f"[init_summary] experiment={experiment_name} "
        f"epoch=0/{total_epochs} "
        f"lr={learning_rate:.6e} "
        f"analysis_sec={analysis_time_sec:.4f}"
    )
    if layer_ratios is not None:
        ratios = ", ".join(f"L{i + 1}={value:.4f}" for i, value in enumerate(layer_ratios))
        summary += f" ratios=[{ratios}]"
    return summary


def format_epoch_summary(
    *,
    experiment_name: str,
    epoch: int,
    total_epochs: int,
    total_loss: float,
    task_loss: float,
    reg_loss: float | None,
    learning_rate: float,
    training_time_sec: float,
    eval_time_sec: float | None,
    eval_loss: float | None,
    eval_accuracy: float | None,
    analysis_time_sec: float,
    reg_enabled: bool,
    layer_ratios: list[float] | None,
    eval_top5_accuracy: float | None = None,
    train_images_per_sec: float | None = None,
    eval_images_per_sec: float | None = None,
    peak_memory_mb: float | None = None,
) -> str:
    reg_text = "n/a" if reg_loss is None else f"{reg_loss:.6f}"
    summary = (
        f"[epoch_summary] experiment={experiment_name} "
        f"epoch={epoch}/{total_epochs} "
        f"total_loss={total_loss:.6f} "
        f"task_loss={task_loss:.6f} "
        f"reg_loss={reg_text} "
        f"reg_active={'yes' if reg_enabled else 'no'} "
        f"lr={learning_rate:.6e} "
        f"train_sec={training_time_sec:.2f} "
        f"eval_sec={'n/a' if eval_time_sec is None else f'{eval_time_sec:.2f}'} "
        f"eval_loss={'n/a' if eval_loss is None else f'{eval_loss:.6f}'} "
        f"eval_acc={'n/a' if eval_accuracy is None else f'{eval_accuracy:.4f}'} "
        f"eval_top5={'n/a' if eval_top5_accuracy is None else f'{eval_top5_accuracy:.4f}'} "
        f"train_img_s={'n/a' if train_images_per_sec is None else f'{train_images_per_sec:.2f}'} "
        f"eval_img_s={'n/a' if eval_images_per_sec is None else f'{eval_images_per_sec:.2f}'} "
        f"peak_mem_mb={'n/a' if peak_memory_mb is None else f'{peak_memory_mb:.1f}'} "
        f"analysis_sec={analysis_time_sec:.4f}"
    )
    if layer_ratios is not None:
        ratios = ", ".join(f"L{i + 1}={value:.4f}" for i, value in enumerate(layer_ratios))
        summary += f" ratios=[{ratios}]"
    return summary


def evaluate_model(
    model: ViTForImageClassification,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float | None, float | None, float]:
    losses: list[float] = []
    correct = 0
    total = 0
    model.eval()
    maybe_sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            losses.append(float(criterion(logits, labels).item()))
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
    maybe_sync(device)
    duration = time.perf_counter() - start
    accuracy = (correct / total) if total > 0 else None
    return (safe_mean(losses) if losses else None), accuracy, duration


def train_vision_experiment(
    config: dict[str, Any],
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    del project_root
    training = config["training"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    regularization = config.get("regularization", {})
    instrumentation = config.get("instrumentation", {})

    set_seed(int(config.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cuda_devices = visible_cuda_device_count() if device.type == "cuda" else 1
    num_classes = int(dataset_config.get("num_classes", 10))
    model = build_vit_model(model_config, num_classes=num_classes)
    if config.get("initialization", {}).get("query_key") == "symmetric":
        apply_symmetric_query_key_initialization(model)
    model.to(device)
    model = maybe_wrap_data_parallel(model, device)

    train_loader, test_loader = build_vision_loaders(dataset_config, training, replica_count=num_cuda_devices)
    grad_accum_steps = int(training.get("gradient_accumulation_steps", 1))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(
            float(training.get("beta1", 0.9)),
            float(training.get("beta2", 0.999)),
        ),
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_optimizer_steps = steps_per_epoch * int(training["max_epochs"])
    warmup_epochs = int(training.get("warmup_epochs", 0))
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_start_factor = float(training.get("warmup_start_factor", 0.033))

    def lr_lambda(current_step: int) -> float:
        if total_optimizer_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            progress = current_step / max(warmup_steps, 1)
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)
        if total_optimizer_steps == warmup_steps:
            return 1.0
        cosine_progress = (current_step - warmup_steps) / max(total_optimizer_steps - warmup_steps, 1)
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    label_smoothing = float(training.get("label_smoothing", 0.0))
    train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    run_paths = prepare_run_paths(output_dir)

    eval_every_epochs = int(training.get("eval_every_epochs", 1))
    intervals = regularization.get("intervals", {}).get("layers", []) if regularization.get("enabled") else []
    reg_schedule = regularization.get("schedule", [])
    lambda_value = float(regularization.get("lambda", 0.0))
    penalty = str(regularization.get("penalty", "linear_hinge"))

    batch_example_pixels, batch_example_labels = next(iter(train_loader))
    batch_example = {
        "pixel_values": batch_example_pixels.to(device),
        "labels": batch_example_labels.to(device),
    }
    task_flops_per_microbatch, flops_method = estimate_model_flops(model, batch_example)
    hidden_size = int(model_config["hidden_size"])
    num_layers = int(model_config["num_hidden_layers"])
    reg_flops_per_microbatch = count_reg_flops(hidden_size, num_layers)
    analysis_flops_per_epoch = count_analysis_flops(hidden_size, num_layers)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(training.get("use_amp", True)))
    use_amp = scaler.is_enabled()

    optimizer.zero_grad(set_to_none=True)

    epoch_metrics: list[dict[str, Any]] = []
    ratio_history: list[dict[str, Any]] = []

    global_microbatches = 0
    global_optimizer_steps = 0
    total_training_time = 0.0
    total_eval_time = 0.0
    total_analysis_time = 0.0
    total_task_flops = 0
    total_reg_flops = 0
    total_analysis_flops = 0

    maybe_sync(device)
    init_analysis_start = time.perf_counter()
    initial_layer_ratios = compute_layer_asymmetry_ratios(model)
    maybe_sync(device)
    init_analysis_time = time.perf_counter() - init_analysis_start
    total_analysis_time += init_analysis_time
    total_analysis_flops += analysis_flops_per_epoch

    ratio_history.append({"epoch": 0, "layer_asymmetry_ratio": [round(v, 8) for v in initial_layer_ratios]})
    epoch_metrics.append(
        {
            "epoch": 0,
            "regularization_active": False,
            "avg_task_loss": None,
            "avg_total_loss": None,
            "avg_reg_loss": None,
            "training_time_sec": 0.0,
            "evaluation_time_sec": None,
            "eval_loss": None,
            "eval_accuracy": None,
            "analysis_time_sec": round(init_analysis_time, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "layer_asymmetry_ratio": [round(v, 8) for v in initial_layer_ratios],
        }
    )
    print(
        format_init_summary(
            experiment_name=config.get("experiment_name", "remote-lab-vision-train"),
            total_epochs=int(training["max_epochs"]),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            analysis_time_sec=init_analysis_time,
            layer_ratios=initial_layer_ratios,
        ),
        flush=True,
    )

    use_tqdm = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, "fileno") else False
    progress = tqdm(
        range(1, int(training["max_epochs"]) + 1),
        desc=config.get("experiment_name", "remote-lab-vision-train"),
        disable=not use_tqdm,
    )

    amp_context = torch.autocast if hasattr(torch, "autocast") else None

    for epoch in progress:
        model.train()
        epoch_total_losses: list[float] = []
        epoch_task_losses: list[float] = []
        epoch_reg_losses: list[float] = []
        reg_enabled = bool(regularization.get("enabled")) and regularization_active(epoch, reg_schedule)
        eval_loss: float | None = None
        eval_accuracy: float | None = None
        eval_time_sec: float | None = None

        maybe_sync(device)
        epoch_train_start = time.perf_counter()

        for microbatch_idx, (pixel_values, labels) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            autocast_context = (
                amp_context(device_type=device.type, dtype=torch.float16, enabled=use_amp)
                if amp_context is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model(pixel_values=pixel_values).logits
                task_loss = train_criterion(logits, labels)
                total_loss = task_loss
                reg_loss_value = 0.0
                if reg_enabled:
                    reg_loss, _ = compute_reg_loss(model, intervals, penalty=penalty)
                    reg_loss_value = float(reg_loss.detach().item())
                    total_loss = total_loss + lambda_value * reg_loss

            scaled_loss = total_loss / grad_accum_steps
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if reg_enabled:
                total_reg_flops += reg_flops_per_microbatch

            if microbatch_idx % grad_accum_steps == 0 or microbatch_idx == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_optimizer_steps += 1

            global_microbatches += 1
            if task_flops_per_microbatch is not None:
                total_task_flops += task_flops_per_microbatch

            epoch_task_losses.append(float(task_loss.detach().item()))
            epoch_total_losses.append(float(total_loss.detach().item()))
            if reg_enabled:
                epoch_reg_losses.append(reg_loss_value)

        maybe_sync(device)
        epoch_train_time = time.perf_counter() - epoch_train_start
        total_training_time += epoch_train_time

        maybe_sync(device)
        analysis_start = time.perf_counter()
        layer_ratios = compute_layer_asymmetry_ratios(model)
        maybe_sync(device)
        analysis_time = time.perf_counter() - analysis_start
        total_analysis_time += analysis_time
        total_analysis_flops += analysis_flops_per_epoch

        if eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            eval_loss, eval_accuracy, eval_duration = evaluate_model(model, test_loader, device, eval_criterion)
            total_eval_time += eval_duration
            eval_time_sec = eval_duration

        ratio_history.append({"epoch": epoch, "layer_asymmetry_ratio": [round(v, 8) for v in layer_ratios]})
        epoch_metrics.append(
            {
                "epoch": epoch,
                "regularization_active": reg_enabled,
                "avg_task_loss": round(safe_mean(epoch_task_losses), 8),
                "avg_total_loss": round(safe_mean(epoch_total_losses), 8),
                "avg_reg_loss": round(safe_mean(epoch_reg_losses), 8) if epoch_reg_losses else None,
                "training_time_sec": round(epoch_train_time, 6),
                "evaluation_time_sec": round(eval_time_sec, 6) if eval_time_sec is not None else None,
                "eval_loss": round(eval_loss, 8) if eval_loss is not None else None,
                "eval_accuracy": round(eval_accuracy, 8) if eval_accuracy is not None else None,
                "analysis_time_sec": round(analysis_time, 6),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "layer_asymmetry_ratio": [round(v, 8) for v in layer_ratios],
            }
        )

        avg_total_loss = safe_mean(epoch_total_losses)
        avg_task_loss = safe_mean(epoch_task_losses)
        avg_reg_loss = safe_mean(epoch_reg_losses) if epoch_reg_losses else None
        if use_tqdm:
            progress.set_postfix(
                loss=f"{avg_total_loss:.4f}",
                reg="on" if reg_enabled else "off",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                acc="n/a" if eval_accuracy is None else f"{eval_accuracy:.3f}",
            )
        print(
            format_epoch_summary(
                experiment_name=config.get("experiment_name", "remote-lab-vision-train"),
                epoch=epoch,
                total_epochs=int(training["max_epochs"]),
                total_loss=avg_total_loss,
                task_loss=avg_task_loss,
                reg_loss=avg_reg_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                training_time_sec=epoch_train_time,
                eval_time_sec=eval_time_sec,
                eval_loss=eval_loss,
                eval_accuracy=eval_accuracy,
                analysis_time_sec=analysis_time,
                reg_enabled=reg_enabled,
                layer_ratios=layer_ratios,
            ),
            flush=True,
        )

    final_eval_loss = next((row["eval_loss"] for row in reversed(epoch_metrics) if row.get("eval_loss") is not None), None)
    final_eval_accuracy = next((row["eval_accuracy"] for row in reversed(epoch_metrics) if row.get("eval_accuracy") is not None), None)
    if final_eval_loss is None or final_eval_accuracy is None:
        final_eval_loss, final_eval_accuracy, eval_duration = evaluate_model(model, test_loader, device, eval_criterion)
        total_eval_time += eval_duration

    unwrap_model(model).save_pretrained(run_paths.model_dir)

    metrics = {
        "experiment_name": config.get("experiment_name"),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_cuda_devices": num_cuda_devices,
        "data_parallel_enabled": bool(device.type == "cuda" and num_cuda_devices > 1),
        "dataset_name": dataset_config.get("name"),
        "epochs": int(training["max_epochs"]),
        "global_microbatches": global_microbatches,
        "global_optimizer_steps": global_optimizer_steps,
        "final_train_loss": epoch_metrics[-1]["avg_total_loss"] if epoch_metrics else None,
        "final_eval_loss": round(float(final_eval_loss), 8) if final_eval_loss is not None else None,
        "final_eval_accuracy": round(float(final_eval_accuracy), 8) if final_eval_accuracy is not None else None,
        "training_time_sec": round(total_training_time, 6),
        "evaluation_time_sec": round(total_eval_time, 6),
        "analysis_time_sec": round(total_analysis_time, 6),
        "task_flops": total_task_flops if instrumentation.get("measure_flops", False) else None,
        "reg_flops": total_reg_flops if instrumentation.get("measure_flops", False) else None,
        "analysis_flops": total_analysis_flops if instrumentation.get("measure_flops", False) else None,
        "reported_training_flops": (
            total_task_flops + total_reg_flops
            if instrumentation.get("exclude_analysis_flops", True)
            else total_task_flops + total_reg_flops + total_analysis_flops
        )
        if instrumentation.get("measure_flops", False)
        else None,
        "reported_training_time_sec": (
            total_training_time if instrumentation.get("exclude_analysis_time", True) else total_training_time + total_analysis_time
        ),
        "flops_estimation_method": flops_method,
    }

    summary = {
        "config": config,
        "metrics": metrics,
        "epoch_metrics_path": str(run_paths.epoch_metrics_path),
        "ratio_history_path": str(run_paths.ratio_path),
        "model_dir": str(run_paths.model_dir),
    }

    write_json(run_paths.metrics_path, metrics)
    write_json(run_paths.epoch_metrics_path, epoch_metrics)
    write_json(run_paths.ratio_path, ratio_history)
    write_json(run_paths.summary_path, summary)
    return summary
