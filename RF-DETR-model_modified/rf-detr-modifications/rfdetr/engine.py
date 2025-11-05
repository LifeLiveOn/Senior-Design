# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import math
import random
from rfdetr.util.misc import NestedTensor
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.datasets.coco import compute_multi_scale_scales
import rfdetr.util.misc as utils
from torch.amp import autocast, GradScaler


def get_autocast_args(args):
    try:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}
    except TypeError:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks=None,
):
    model.train()
    criterion.train()

    scaler = GradScaler('cuda', enabled=args.amp)
    start_steps = epoch * num_training_steps_per_epoch

    print(f"Grad accum steps: {args.grad_accum_steps}")
    print(f"Total batch size: {batch_size * utils.get_world_size()}")
    print(f"Length of DataLoader: {len(data_loader)}")

    sub_batch_size = batch_size // args.grad_accum_steps
    optimizer.zero_grad()

    pbar = tqdm(total=len(data_loader), desc=f"Epoch {epoch}", ncols=100)

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        it = start_steps + data_iter_step

        if "dp" in schedules:
            update_fn = model.module.update_drop_path if args.distributed else model.update_drop_path
            update_fn(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            update_fn = model.module.update_dropout if args.distributed else model.update_dropout
            update_fn(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(
                args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(it)
            scale = random.choice(scales)
            with torch.inference_mode():
                samples.tensors = F.interpolate(
                    samples.tensors, size=scale, mode='bilinear', align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(
                    1).float(), size=scale, mode='nearest').squeeze(1).bool()

        for i in range(args.grad_accum_steps):
            start_idx, end_idx = i * sub_batch_size, (i + 1) * sub_batch_size
            new_samples = NestedTensor(samples.tensors[start_idx:end_idx],
                                       samples.mask[start_idx:end_idx]).to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets[start_idx:end_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                loss = sum((1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                           for k in loss_dict if k in weight_dict)

            scaler.scale(loss).backward()

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        if ema_m and epoch >= 0:
            ema_m.update(model)

        reduced = utils.reduce_dict(loss_dict)
        weighted = {k: v * weight_dict[k]
                    for k, v in reduced.items() if k in weight_dict}
        total_loss = sum(weighted.values()).item()

        pbar.set_postfix({
            "loss": f"{total_loss:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        pbar.update(1)

    pbar.close()
    return {"loss": total_loss}


def coco_extended_metrics(coco_eval):
    """
    Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
    """

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    iou50_idx, area_idx, maxdet_idx = (
        int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, 2)

    P = coco_eval.eval["precision"]
    S = coco_eval.eval["scores"]

    prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan

    f1_cls = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
    f1_macro = np.nanmean(f1_cls, axis=1)

    best_j = int(f1_macro.argmax())

    macro_precision = float(np.nanmean(prec[best_j]))
    macro_recall = float(rec_thrs[best_j])
    macro_f1 = float(f1_macro[best_j])

    score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
    score_vec[prec_raw[best_j] < 0] = np.nan
    score_thr = float(np.nanmean(score_vec))

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"]
                      for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]
        valid = p_slice > -1
        ap_50_95 = float(p_slice[valid].mean()
                         ) if valid.any() else float("nan")
        ap_50 = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()
                      ) if (p_slice[iou50_idx] > -1).any() else float("nan")

        pc = float(prec[best_j, k]) if prec_raw[best_j,
                                                k] > -1 else float("nan")
        rc = macro_recall

        # Doing to this to filter out dataset class
        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
            continue

        per_class.append({
            "class": cat_id_to_name[int(cid)],
            "map@50:95": ap_50_95,
            "map@50": ap_50,
            "precision": pc,
            "recall": rc,
        })

    per_class.append({
        "class": "all",
        "map@50:95": map_50_95,
        "map@50": map_50,
        "precision": macro_precision,
        "recall": macro_recall,
    })

    return {
        "class_map": per_class,
        "map": map_50,
        "precision": macro_precision,
        "recall": macro_recall
    }


def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    criterion.eval()
    if args.fp16_eval:
        model.half()

    coco_evaluator = CocoEvaluator(
        base_ds, ("bbox",) if not args.segmentation_head else ("bbox", "segm"))
    pbar = tqdm(total=len(data_loader), desc="Evaluating", ncols=100)

    total_loss = 0.0
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        reduced = utils.reduce_dict(loss_dict)
        weighted = {k: v * weight_dict[k]
                    for k, v in reduced.items() if k in weight_dict}
        batch_loss = sum(weighted.values()).item()
        total_loss += batch_loss

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = postprocess(outputs, orig_sizes)
        res = {t["image_id"].item(): r for t, r in zip(targets, results)}
        coco_evaluator.update(res)

        pbar.set_postfix({"val_loss": f"{batch_loss:.3f}"})
        pbar.update(1)

    pbar.close()

    # Finish evaluation and collect metrics
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Prepare return stats
    stats = {"val_loss": total_loss / len(data_loader)}

    # Extract metrics safely
    if "bbox" in coco_evaluator.coco_eval:
        coco_eval = coco_evaluator.coco_eval["bbox"]
        stats["coco_eval_bbox"] = coco_eval.stats.tolist()
        # optional: generate additional metrics
        extended = coco_extended_metrics(coco_eval)
        stats.update({
            "map_regular": extended["map"],
            "precision": extended["precision"],
            "recall": extended["recall"],
            # compatible with older RF-DETR
            "results_json": {"class_map": extended["class_map"]},
        })
    if "segm" in coco_evaluator.coco_eval:
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
