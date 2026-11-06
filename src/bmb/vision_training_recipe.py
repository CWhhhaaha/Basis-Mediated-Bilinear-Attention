from __future__ import annotations

import copy
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import ViTForImageClassification

from bmb.attention_accounting import attention_forward_accounting
from bmb.vision_training import (
    RunPaths,
    apply_symmetric_query_key_initialization,
    build_vision_loaders,
    build_vit_model,
    compute_layer_asymmetry_ratios,
    compute_reg_loss,
    count_analysis_flops,
    count_reg_flops,
    estimate_model_flops,
    format_epoch_summary,
    format_init_summary,
    maybe_sync,
    maybe_wrap_data_parallel,
    prepare_run_paths,
    regularization_active,
    safe_mean,
    set_seed,
    unwrap_model,
    visible_cuda_device_count,
    write_json,
)


class ModelEma:
    def __init__(self, model: ViTForImageClassification, decay: float, device: torch.device) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).to(device)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: ViTForImageClassification) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(value):
                value.copy_(model_value)
            else:
                value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off_value = smoothing / max(num_classes - 1, 1)
    on_value = 1.0 - smoothing
    result = torch.full((labels.size(0), num_classes), off_value, device=labels.device, dtype=torch.float32)
    result.scatter_(1, labels.unsqueeze(1), on_value)
    return result


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def rand_bbox(size: tuple[int, int, int, int], lam: float) -> tuple[int, int, int, int]:
    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = torch.randint(0, width, (1,)).item()
    cy = torch.randint(0, height, (1,)).item()

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_mixup_or_cutmix(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    switch_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0
    if not use_mixup and not use_cutmix:
        return pixel_values, smooth_one_hot(labels, num_classes, label_smoothing)

    perm = torch.randperm(labels.size(0), device=labels.device)
    labels_a = smooth_one_hot(labels, num_classes, label_smoothing)
    labels_b = smooth_one_hot(labels[perm], num_classes, label_smoothing)

    choose_cutmix = use_cutmix and (not use_mixup or torch.rand(1).item() < switch_prob)
    if choose_cutmix:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        x1, y1, x2, y2 = rand_bbox(tuple(pixel_values.shape), lam)
        mixed = pixel_values.clone()
        mixed[:, :, y1:y2, x1:x2] = pixel_values[perm, :, y1:y2, x1:x2]
        box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        adjusted_lam = 1.0 - (box_area / float(pixel_values.size(-1) * pixel_values.size(-2)))
        targets = adjusted_lam * labels_a + (1.0 - adjusted_lam) * labels_b
        return mixed, targets

    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
    mixed = lam * pixel_values + (1.0 - lam) * pixel_values[perm]
    targets = lam * labels_a + (1.0 - lam) * labels_b
    return mixed, targets


def evaluate_model_recipe(
    model: ViTForImageClassification,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float | None, float | None, float | None, float]:
    losses: list[float] = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    model.eval()
    maybe_sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(pixel_values=pixel_values).logits
            losses.append(float(criterion(logits, labels).item()))
            max_k = min(5, logits.size(-1))
            predictions = logits.topk(max_k, dim=-1).indices
            correct_top1 += int((predictions[:, 0] == labels).sum().item())
            correct_top5 += int((predictions == labels.unsqueeze(1)).any(dim=1).sum().item())
            total += int(labels.numel())
    maybe_sync(device)
    duration = time.perf_counter() - start
    top1 = (correct_top1 / total) if total > 0 else None
    top5 = (correct_top5 / total) if total > 0 else None
    return (safe_mean(losses) if losses else None), top1, top5, duration


def bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def current_cuda_peak_memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "peak_cuda_memory_allocated_mb": None,
            "peak_cuda_memory_reserved_mb": None,
        }
    device_count = visible_cuda_device_count()
    allocated_values = [bytes_to_mb(torch.cuda.max_memory_allocated(idx)) for idx in range(device_count)]
    reserved_values = [bytes_to_mb(torch.cuda.max_memory_reserved(idx)) for idx in range(device_count)]
    allocated_values = [value for value in allocated_values if value is not None]
    reserved_values = [value for value in reserved_values if value is not None]
    return {
        "peak_cuda_memory_allocated_mb": max(allocated_values) if allocated_values else None,
        "peak_cuda_memory_reserved_mb": max(reserved_values) if reserved_values else None,
    }


def count_attention_parameters(model: ViTForImageClassification) -> int:
    base_model = unwrap_model(model)
    return sum(param.numel() for name, param in base_model.named_parameters() if ".attention." in name)


def count_qk_score_parameters(model: ViTForImageClassification) -> int:
    base_model = unwrap_model(model)
    total = 0
    for layer in base_model.vit.encoder.layer:
        attention = layer.attention.attention
        if hasattr(attention, "query") and hasattr(attention, "key"):
            total += sum(param.numel() for param in attention.query.parameters())
            total += sum(param.numel() for param in attention.key.parameters())
        else:
            for attr in (
                "basis",
                "core",
                "head_residual",
                "u_factor",
                "v_factor",
                "share",
                "query_priv",
                "key_priv",
                "query_key",
                "q_a",
                "q_b",
                "k_a",
                "k_b",
            ):
                module_or_param = getattr(attention, attr, None)
                if module_or_param is None:
                    continue
                if isinstance(module_or_param, nn.Parameter):
                    total += module_or_param.numel()
                elif isinstance(module_or_param, nn.Module):
                    total += sum(param.numel() for param in module_or_param.parameters())
    return total


def theoretical_attention_summary(model_config: dict[str, Any]) -> dict[str, float | int | str | None]:
    variant = str(model_config.get("attention_variant", "standard"))
    summary = attention_forward_accounting(
        variant,
        model_dim=int(model_config["hidden_size"]),
        num_heads=int(model_config["num_attention_heads"]),
        num_layers=int(model_config["num_hidden_layers"]),
        tokens=(int(model_config.get("image_size", 32)) // int(model_config.get("patch_size", 4))) ** 2 + 1,
        rank=model_config.get("latent_rank", model_config.get("low_rank")),
        factor_rank=model_config.get("latent_factor_rank"),
        shared_dim=model_config.get("shared_qk_dim"),
    )
    qk_params = summary["qk_params_per_layer"]
    attention_params = summary["attention_params_per_layer"]
    qk_flops = summary["qk_flops_per_layer"]
    attention_flops = summary["attention_flops_per_layer"]
    baseline_qk_params = summary["baseline_qk_params_total"] // summary["num_layers"]
    baseline_attention_params = summary["baseline_attention_params_total"] // summary["num_layers"]
    baseline_qk_flops = summary["baseline_qk_flops_total"] // summary["num_layers"]
    baseline_attention_flops = summary["baseline_attention_flops_total"] // summary["num_layers"]
    latent_rank = model_config.get("latent_rank") if variant in {"layer_symmetric_latent", "layer_uv_latent", "layer_bbt"} else None
    latent_factor_rank = model_config.get("latent_factor_rank", latent_rank) if variant == "layer_uv_latent" else None
    return {
        "attention_variant": variant,
        "latent_rank": latent_rank,
        "latent_factor_rank": latent_factor_rank,
        "hidden_size": summary["model_dim"],
        "num_layers": summary["num_layers"],
        "num_heads": summary["num_heads"],
        "head_dim": summary["head_dim"],
        "tokens_per_example": summary["tokens"],
        "qk_weight_params_per_layer": qk_params,
        "qk_weight_params_total": summary["qk_params_total"],
        "baseline_qk_weight_params_total": summary["baseline_qk_params_total"],
        "qk_weight_param_reduction_pct": round(100.0 * (baseline_qk_params - qk_params) / baseline_qk_params, 6),
        "attention_weight_params_per_layer": attention_params,
        "attention_weight_params_total": summary["attention_params_total"],
        "baseline_attention_weight_params_total": summary["baseline_attention_params_total"],
        "attention_weight_param_reduction_pct": round(100.0 * (baseline_attention_params - attention_params) / baseline_attention_params, 6),
        "qk_flops_per_example_per_layer": qk_flops,
        "qk_flops_per_example_total": summary["qk_flops_total"],
        "baseline_qk_flops_per_example_total": summary["baseline_qk_flops_total"],
        "qk_flops_reduction_pct": round(100.0 * (baseline_qk_flops - qk_flops) / baseline_qk_flops, 6),
        "attention_flops_per_example_per_layer": attention_flops,
        "attention_flops_per_example_total": summary["attention_flops_total"],
        "baseline_attention_flops_per_example_total": summary["baseline_attention_flops_total"],
        "attention_flops_reduction_pct": round(100.0 * (baseline_attention_flops - attention_flops) / baseline_attention_flops, 6),
    }


def entropy_effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> float:
    singular_values = torch.linalg.svdvals(matrix.float())
    total = singular_values.sum().clamp_min(eps)
    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())


def off_diagonal_mean(matrix: torch.Tensor) -> float | None:
    size = matrix.size(0)
    if size <= 1:
        return None
    mask = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].mean().item())


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def summarize_bmb_diagnostics(model: ViTForImageClassification) -> dict[str, Any] | None:
    per_layer: list[dict[str, Any]] = []
    erank_b_values: list[float] = []
    erank_m_values: list[float] = []
    head_cos_values: list[float] = []

    with torch.no_grad():
        base_model = unwrap_model(model)
        for layer_idx, layer in enumerate(base_model.vit.encoder.layer, start=1):
            attention = layer.attention.attention
            if not hasattr(attention, "basis") or not hasattr(attention, "head_matrices"):
                continue

            basis = attention.basis.weight.detach().float().transpose(0, 1).cpu()
            head_matrices = attention.head_matrices().detach().float().cpu()

            erank_b = entropy_effective_rank(basis)
            erank_m_per_head = [entropy_effective_rank(head_matrices[head_idx]) for head_idx in range(head_matrices.size(0))]

            flattened = F.normalize(head_matrices.reshape(head_matrices.size(0), -1), dim=1)
            similarity = flattened @ flattened.transpose(0, 1)
            head_cosine_mean = off_diagonal_mean(similarity)

            erank_b_values.append(erank_b)
            erank_m_values.extend(erank_m_per_head)
            if head_cosine_mean is not None:
                head_cos_values.append(head_cosine_mean)

            per_layer.append(
                {
                    "layer": layer_idx,
                    "effective_rank_B": round(erank_b, 6),
                    "effective_rank_M_mean": round(sum(erank_m_per_head) / len(erank_m_per_head), 6),
                    "effective_rank_M_min": round(min(erank_m_per_head), 6),
                    "effective_rank_M_max": round(max(erank_m_per_head), 6),
                    "head_M_cosine_similarity_mean": round(head_cosine_mean, 6) if head_cosine_mean is not None else None,
                }
            )

    if not per_layer:
        return None

    return {
        "effective_rank_B_final": summarize_values(erank_b_values),
        "effective_rank_M_final": summarize_values(erank_m_values),
        "head_M_cosine_similarity_final": summarize_values(head_cos_values),
        "per_layer": per_layer,
    }


def best_metric(rows: list[dict[str, Any]], key: str) -> tuple[float | None, int | None]:
    best_value: float | None = None
    best_epoch: int | None = None
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        numeric = float(value)
        if best_value is None or numeric > best_value:
            best_value = numeric
            best_epoch = int(row["epoch"])
    return best_value, best_epoch


def train_vision_recipe_experiment(
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
    augmentation = config.get("augmentation", {})
    ema_config = config.get("ema", {})

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
    train_dataset_size = len(train_loader.dataset)
    eval_dataset_size = len(test_loader.dataset)
    grad_accum_steps = int(training.get("gradient_accumulation_steps", 1))
    parameter_summary = {
        "total_params": sum(param.numel() for param in model.parameters()),
        "trainable_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "attention_params": count_attention_parameters(model),
        "qk_score_params": count_qk_score_parameters(model),
    }
    attention_theory_summary = theoretical_attention_summary(model_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.999))),
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
    eval_criterion = nn.CrossEntropyLoss()
    run_paths: RunPaths = prepare_run_paths(output_dir)

    eval_every_epochs = int(training.get("eval_every_epochs", 1))
    intervals = regularization.get("intervals", {}).get("layers", []) if regularization.get("enabled") else []
    reg_schedule = regularization.get("schedule", [])
    lambda_value = float(regularization.get("lambda", 0.0))
    penalty = str(regularization.get("penalty", "linear_hinge"))

    mixup_alpha = float(augmentation.get("mixup_alpha", 0.0))
    cutmix_alpha = float(augmentation.get("cutmix_alpha", 0.0))
    switch_prob = float(augmentation.get("switch_prob", 0.5))

    batch_example_pixels, batch_example_labels = next(iter(train_loader))
    batch_example = {"pixel_values": batch_example_pixels.to(device), "labels": batch_example_labels.to(device)}
    task_flops_per_microbatch, flops_method = estimate_model_flops(model, batch_example)
    hidden_size = int(model_config["hidden_size"])
    num_layers = int(model_config["num_hidden_layers"])
    reg_flops_per_microbatch = count_reg_flops(hidden_size, num_layers)
    analysis_flops_per_epoch = count_analysis_flops(hidden_size, num_layers)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(training.get("use_amp", True)))
    use_amp = scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    ema_enabled = bool(ema_config.get("enabled", False))
    ema_model = ModelEma(model, decay=float(ema_config.get("decay", 0.99998)), device=device) if ema_enabled else None
    ema_update_every_steps = int(ema_config.get("update_every_steps", 32))

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
    measure_layer_ratios = bool(instrumentation.get("measure_layer_ratios", False))

    initial_layer_ratios: list[float] | None = None
    init_analysis_time = 0.0
    if measure_layer_ratios:
        maybe_sync(device)
        init_analysis_start = time.perf_counter()
        initial_layer_ratios = compute_layer_asymmetry_ratios(model)
        maybe_sync(device)
        init_analysis_time = time.perf_counter() - init_analysis_start
        total_analysis_time += init_analysis_time
        total_analysis_flops += analysis_flops_per_epoch

    if initial_layer_ratios is not None:
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
            "eval_loss_raw": None,
            "eval_accuracy_raw": None,
            "eval_top5_accuracy_raw": None,
            "eval_loss_ema": None,
            "eval_accuracy_ema": None,
            "eval_top5_accuracy_ema": None,
            "analysis_time_sec": round(init_analysis_time, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_images_per_sec": None,
            "eval_images_per_sec": None,
            "train_peak_cuda_memory_allocated_mb": None,
            "train_peak_cuda_memory_reserved_mb": None,
            "eval_peak_cuda_memory_allocated_mb": None,
            "eval_peak_cuda_memory_reserved_mb": None,
            "peak_cuda_memory_allocated_mb": None,
            "peak_cuda_memory_reserved_mb": None,
            "layer_asymmetry_ratio": (
                [round(v, 8) for v in initial_layer_ratios] if initial_layer_ratios is not None else None
            ),
        }
    )
    print(
        format_init_summary(
            experiment_name=config.get("experiment_name", "remote-lab-vision-recipe-train"),
            total_epochs=int(training["max_epochs"]),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            analysis_time_sec=init_analysis_time,
            layer_ratios=initial_layer_ratios,
        ),
        flush=True,
    )

    use_tqdm = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, "fileno") else False
    progress = tqdm(range(1, int(training["max_epochs"]) + 1), desc=config.get("experiment_name", "remote-lab-vision-recipe-train"), disable=not use_tqdm)
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
        eval_loss_raw: float | None = None
        eval_accuracy_raw: float | None = None
        eval_top5_accuracy_raw: float | None = None
        eval_loss_ema: float | None = None
        eval_accuracy_ema: float | None = None
        eval_top5_accuracy_ema: float | None = None
        eval_top5_accuracy: float | None = None
        eval_images_per_sec: float | None = None
        eval_peak_memory = {
            "peak_cuda_memory_allocated_mb": None,
            "peak_cuda_memory_reserved_mb": None,
        }

        maybe_sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_train_start = time.perf_counter()

        for microbatch_idx, (pixel_values, labels) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed_pixels, soft_targets = apply_mixup_or_cutmix(
                pixel_values,
                labels,
                num_classes=num_classes,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                switch_prob=switch_prob,
            )

            autocast_context = (
                amp_context(device_type=device.type, dtype=torch.float16, enabled=use_amp)
                if amp_context is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model(pixel_values=mixed_pixels).logits
                task_loss = soft_target_cross_entropy(logits, soft_targets)
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
                if ema_model is not None and global_optimizer_steps % max(ema_update_every_steps, 1) == 0:
                    ema_model.update(model)

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
        train_images_per_sec = train_dataset_size / epoch_train_time if epoch_train_time > 0 else None
        train_peak_memory = current_cuda_peak_memory(device)

        layer_ratios: list[float] | None = None
        analysis_time = 0.0
        if measure_layer_ratios:
            maybe_sync(device)
            analysis_start = time.perf_counter()
            layer_ratios = compute_layer_asymmetry_ratios(model)
            maybe_sync(device)
            analysis_time = time.perf_counter() - analysis_start
            total_analysis_time += analysis_time
            total_analysis_flops += analysis_flops_per_epoch

        if eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            eval_loss_raw, eval_accuracy_raw, eval_top5_accuracy_raw, raw_duration = evaluate_model_recipe(
                model, test_loader, device, eval_criterion
            )
            total_eval_time += raw_duration
            eval_time_sec = raw_duration
            if ema_model is not None:
                eval_loss_ema, eval_accuracy_ema, eval_top5_accuracy_ema, ema_duration = evaluate_model_recipe(
                    ema_model.module, test_loader, device, eval_criterion
                )
                total_eval_time += ema_duration
                eval_time_sec += ema_duration
                eval_loss = eval_loss_ema
                eval_accuracy = eval_accuracy_ema
                eval_top5_accuracy = eval_top5_accuracy_ema
            else:
                eval_loss = eval_loss_raw
                eval_accuracy = eval_accuracy_raw
                eval_top5_accuracy = eval_top5_accuracy_raw
            eval_images_per_sec = eval_dataset_size / eval_time_sec if eval_time_sec and eval_time_sec > 0 else None
            eval_peak_memory = current_cuda_peak_memory(device)

        peak_allocated_values = [
            value
            for value in (
                train_peak_memory["peak_cuda_memory_allocated_mb"],
                eval_peak_memory["peak_cuda_memory_allocated_mb"],
            )
            if value is not None
        ]
        peak_reserved_values = [
            value
            for value in (
                train_peak_memory["peak_cuda_memory_reserved_mb"],
                eval_peak_memory["peak_cuda_memory_reserved_mb"],
            )
            if value is not None
        ]
        peak_cuda_memory_allocated_mb = max(peak_allocated_values) if peak_allocated_values else None
        peak_cuda_memory_reserved_mb = max(peak_reserved_values) if peak_reserved_values else None

        if layer_ratios is not None:
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
                "eval_loss_raw": round(eval_loss_raw, 8) if eval_loss_raw is not None else None,
                "eval_accuracy_raw": round(eval_accuracy_raw, 8) if eval_accuracy_raw is not None else None,
                "eval_top5_accuracy_raw": round(eval_top5_accuracy_raw, 8) if eval_top5_accuracy_raw is not None else None,
                "eval_loss_ema": round(eval_loss_ema, 8) if eval_loss_ema is not None else None,
                "eval_accuracy_ema": round(eval_accuracy_ema, 8) if eval_accuracy_ema is not None else None,
                "eval_top5_accuracy_ema": round(eval_top5_accuracy_ema, 8) if eval_top5_accuracy_ema is not None else None,
                "eval_top5_accuracy": round(eval_top5_accuracy, 8) if eval_top5_accuracy is not None else None,
                "analysis_time_sec": round(analysis_time, 6),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_images_per_sec": round(train_images_per_sec, 6) if train_images_per_sec is not None else None,
                "eval_images_per_sec": round(eval_images_per_sec, 6) if eval_images_per_sec is not None else None,
                "train_peak_cuda_memory_allocated_mb": (
                    round(train_peak_memory["peak_cuda_memory_allocated_mb"], 6)
                    if train_peak_memory["peak_cuda_memory_allocated_mb"] is not None
                    else None
                ),
                "train_peak_cuda_memory_reserved_mb": (
                    round(train_peak_memory["peak_cuda_memory_reserved_mb"], 6)
                    if train_peak_memory["peak_cuda_memory_reserved_mb"] is not None
                    else None
                ),
                "eval_peak_cuda_memory_allocated_mb": (
                    round(eval_peak_memory["peak_cuda_memory_allocated_mb"], 6)
                    if eval_peak_memory["peak_cuda_memory_allocated_mb"] is not None
                    else None
                ),
                "eval_peak_cuda_memory_reserved_mb": (
                    round(eval_peak_memory["peak_cuda_memory_reserved_mb"], 6)
                    if eval_peak_memory["peak_cuda_memory_reserved_mb"] is not None
                    else None
                ),
                "peak_cuda_memory_allocated_mb": (
                    round(peak_cuda_memory_allocated_mb, 6) if peak_cuda_memory_allocated_mb is not None else None
                ),
                "peak_cuda_memory_reserved_mb": (
                    round(peak_cuda_memory_reserved_mb, 6) if peak_cuda_memory_reserved_mb is not None else None
                ),
                "layer_asymmetry_ratio": [round(v, 8) for v in layer_ratios] if layer_ratios is not None else None,
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
                experiment_name=config.get("experiment_name", "remote-lab-vision-recipe-train"),
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
                eval_top5_accuracy=eval_top5_accuracy,
                train_images_per_sec=train_images_per_sec,
                eval_images_per_sec=eval_images_per_sec,
                peak_memory_mb=peak_cuda_memory_allocated_mb,
            ),
            flush=True,
        )

    final_row = epoch_metrics[-1] if epoch_metrics else {}
    best_top1, best_top1_epoch = best_metric(epoch_metrics, "eval_accuracy")
    best_top5, best_top5_epoch = best_metric(epoch_metrics, "eval_top5_accuracy")
    best_top1_raw, best_top1_raw_epoch = best_metric(epoch_metrics, "eval_accuracy_raw")
    best_top5_raw, best_top5_raw_epoch = best_metric(epoch_metrics, "eval_top5_accuracy_raw")
    bmb_diagnostics = summarize_bmb_diagnostics(model)
    unwrap_model(model).save_pretrained(run_paths.model_dir)
    if ema_model is not None:
        ema_model.module.save_pretrained(run_paths.model_dir / "ema")

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
        "final_train_loss": final_row.get("avg_total_loss"),
        "final_eval_loss": final_row.get("eval_loss"),
        "final_eval_accuracy": final_row.get("eval_accuracy"),
        "final_eval_top5_accuracy": final_row.get("eval_top5_accuracy"),
        "final_eval_loss_raw": final_row.get("eval_loss_raw"),
        "final_eval_accuracy_raw": final_row.get("eval_accuracy_raw"),
        "final_eval_top5_accuracy_raw": final_row.get("eval_top5_accuracy_raw"),
        "final_eval_loss_ema": final_row.get("eval_loss_ema"),
        "final_eval_accuracy_ema": final_row.get("eval_accuracy_ema"),
        "final_eval_top5_accuracy_ema": final_row.get("eval_top5_accuracy_ema"),
        "best_eval_accuracy": round(best_top1, 8) if best_top1 is not None else None,
        "best_eval_accuracy_epoch": best_top1_epoch,
        "best_eval_top5_accuracy": round(best_top5, 8) if best_top5 is not None else None,
        "best_eval_top5_accuracy_epoch": best_top5_epoch,
        "best_eval_accuracy_raw": round(best_top1_raw, 8) if best_top1_raw is not None else None,
        "best_eval_accuracy_raw_epoch": best_top1_raw_epoch,
        "best_eval_top5_accuracy_raw": round(best_top5_raw, 8) if best_top5_raw is not None else None,
        "best_eval_top5_accuracy_raw_epoch": best_top5_raw_epoch,
        "training_time_sec": round(total_training_time, 6),
        "evaluation_time_sec": round(total_eval_time, 6),
        "analysis_time_sec": round(total_analysis_time, 6),
        "mean_train_images_per_sec": round(
            safe_mean([float(row["train_images_per_sec"]) for row in epoch_metrics if row.get("train_images_per_sec") is not None]),
            6,
        ),
        "mean_eval_images_per_sec": round(
            safe_mean([float(row["eval_images_per_sec"]) for row in epoch_metrics if row.get("eval_images_per_sec") is not None]),
            6,
        ),
        "peak_cuda_memory_allocated_mb": round(
            max(
                [float(row["peak_cuda_memory_allocated_mb"]) for row in epoch_metrics if row.get("peak_cuda_memory_allocated_mb") is not None],
                default=0.0,
            ),
            6,
        )
        if device.type == "cuda"
        else None,
        "peak_cuda_memory_reserved_mb": round(
            max(
                [float(row["peak_cuda_memory_reserved_mb"]) for row in epoch_metrics if row.get("peak_cuda_memory_reserved_mb") is not None],
                default=0.0,
            ),
            6,
        )
        if device.type == "cuda"
        else None,
        "parameter_summary": parameter_summary,
        "attention_theory_summary": attention_theory_summary,
        "bmb_diagnostics": bmb_diagnostics,
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
        "ema_enabled": ema_enabled,
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
