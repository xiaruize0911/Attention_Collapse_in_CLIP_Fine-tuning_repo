"""
run_all_experiments.py - Complete experiment pipeline for CLIP Attention Structural Collapse
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time
import datetime
import copy
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.model import CLIPClassifier, create_lora_model, get_pretrained_model, count_parameters
from src.dataset import (load_eurosat, load_oxford_pets, create_fixed_eval_subset,
                          get_dataloader, get_clip_transform, load_cifar100, load_flowers102)
from src.metrics import compute_all_metrics, compute_attention_rollout
from src.regularizer import AttentionPreservationRegularizer, EntropyFloorRegularizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
OUTPUT_DIR = Path("outputs")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics_on_eval(model, eval_loader, device):
    """Compute attention metrics on fixed eval set."""
    model.eval()
    all_metrics_list = []
    
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            _, attentions = model(images, output_attentions=True)
            metrics = compute_all_metrics(attentions)
            all_metrics_list.append(metrics)
    
    # Average over batches
    avg = {}
    for key in all_metrics_list[0]:
        if isinstance(all_metrics_list[0][key], list):
            avg[key] = [np.mean([m[key][i] for m in all_metrics_list]) 
                       for i in range(len(all_metrics_list[0][key]))]
        else:
            avg[key] = np.mean([m[key] for m in all_metrics_list])
    return avg


def evaluate_accuracy(model, loader, device):
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images, output_attentions=False)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total if total > 0 else 0.0


def train_one_experiment(config, model, train_loader, val_loader, eval_loader,
                          writer, experiment_id, regularizer=None):
    """
    Train one experiment with full logging.
    Returns: dict with results and metrics history
    """
    model = model.to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Scheduler with warmup
    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(0.05 * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    
    # Results tracking
    history = {
        'train_loss': [], 'train_acc': [], 'val_acc': [],
        'attention_metrics': [], 'steps': [], 'epochs': []
    }
    
    checkpoint_dir = OUTPUT_DIR / "checkpoints" / experiment_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "logs" / "train" / f"{experiment_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    step = 0
    best_val_acc = 0.0
    
    # Baseline metrics before training
    baseline_metrics = compute_metrics_on_eval(model, eval_loader, DEVICE)
    history['baseline_metrics'] = baseline_metrics
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_id}")
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    print(f"Config: {config}")
    print(f"Baseline entropy mean: {baseline_metrics['entropy_mean']:.4f}")
    print(f"{'='*60}")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", 
                    leave=False, ncols=100)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, attentions = model(images, output_attentions=True)
                loss = criterion(logits, labels)
                
                # Regularization
                if regularizer is not None:
                    if isinstance(regularizer, AttentionPreservationRegularizer):
                        reg_loss = regularizer.compute(images, attentions)
                    elif isinstance(regularizer, EntropyFloorRegularizer):
                        reg_loss = regularizer.compute(attentions)
                    else:
                        reg_loss = 0.0
                    loss = loss + reg_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Stats
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            epoch_total += labels.size(0)
            epoch_correct += predicted.eq(labels).sum().item()
            
            # Log every N steps
            log_interval = max(1, len(train_loader) // 5)  # ~5 times per epoch
            if step % log_interval == 0:
                train_acc = epoch_correct / max(epoch_total, 1)
                
                # Compute attention metrics on fixed eval set
                attn_metrics = compute_metrics_on_eval(model, eval_loader, DEVICE)
                model.train()  # Switch back to train mode
                
                # TensorBoard logging
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/accuracy', train_acc, step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
                
                for layer_idx in range(12):
                    writer.add_scalar(f'attention/entropy_layer_{layer_idx}',
                                     attn_metrics['entropy_per_layer'][layer_idx], step)
                    writer.add_scalar(f'attention/erf95_layer_{layer_idx}',
                                     attn_metrics['erf95_per_layer'][layer_idx], step)
                    writer.add_scalar(f'attention/gini_layer_{layer_idx}',
                                     attn_metrics['gini_per_layer'][layer_idx], step)
                
                writer.add_scalar('attention/entropy_mean', attn_metrics['entropy_mean'], step)
                writer.add_scalar('attention/erf95_mean', attn_metrics['erf95_mean'], step)
                writer.add_scalar('attention/gini_mean', attn_metrics['gini_mean'], step)
                writer.add_scalar('attention/head_diversity_mean', attn_metrics['head_diversity_mean'], step)
                
                # Record history
                history['steps'].append(step)
                history['train_loss'].append(loss.item())
                history['train_acc'].append(train_acc)
                history['attention_metrics'].append(attn_metrics)
                
                # JSONL log
                log_entry = {
                    "step": step, "epoch": epoch + batch_idx / len(train_loader),
                    "train_loss": loss.item(), "train_accuracy": train_acc,
                    "attention_metrics": attn_metrics,
                    "lr": optimizer.param_groups[0]['lr'],
                    "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                }
                with open(log_path, 'a') as f:
                    json.dump(log_entry, f)
                    f.write('\n')
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{epoch_correct/max(epoch_total,1):.3f}'})
            step += 1
        
        # End of epoch evaluation
        val_acc = evaluate_accuracy(model, val_loader, DEVICE)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch)
        
        writer.add_scalar('eval/accuracy', val_acc, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch, 'step': step,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc, 'config': config
            }, checkpoint_dir / 'best_model.pth')
        
        train_acc = epoch_correct / max(epoch_total, 1)
        print(f"  Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
              f"entropy={history['attention_metrics'][-1]['entropy_mean']:.4f}")
    
    # Final metrics
    final_metrics = compute_metrics_on_eval(model, eval_loader, DEVICE)
    history['final_metrics'] = final_metrics
    history['best_val_acc'] = best_val_acc
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config, 'history': history,
    }, checkpoint_dir / 'final_model.pth')
    
    # Save history
    metrics_path = OUTPUT_DIR / "metrics" / f"{experiment_id}_history.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    writer.flush()
    return history


def run_baseline_analysis():
    """E1: Baseline - Analyze pretrained model attention."""
    print("\n" + "="*70)
    print("E1: BASELINE ANALYSIS - Pretrained CLIP Attention Statistics")
    print("="*70)
    
    set_seed(SEED)
    
    # Load model
    model = CLIPClassifier(num_classes=10).to(DEVICE)
    model.eval()
    
    # Load EuroSAT for eval images
    train_dataset, test_dataset, num_classes, class_names = load_eurosat(cache_dir="./data")
    eval_subset, eval_indices = create_fixed_eval_subset(test_dataset, num_samples=200, 
                                                          num_classes=num_classes, seed=SEED)
    eval_loader = get_dataloader(eval_subset, batch_size=32, shuffle=False, num_workers=2)
    
    # Save eval indices
    indices_path = OUTPUT_DIR / "metrics" / "fixed_eval_indices.json"
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(indices_path, 'w') as f:
        json.dump(eval_indices, f)
    
    # Compute baseline metrics
    tb_dir = OUTPUT_DIR / "logs" / "tensorboard" / "E1_baseline"
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    baseline_metrics = compute_metrics_on_eval(model, eval_loader, DEVICE)
    
    # Detailed per-layer, per-head analysis
    per_layer_per_head = {'entropy': [], 'erf95': [], 'gini': []}
    
    all_attentions = []
    with torch.no_grad():
        for images, _ in eval_loader:
            images = images.to(DEVICE)
            _, attentions = model(images, output_attentions=True)
            all_attentions.append(attentions)
    
    # Compute per-head metrics
    for layer_idx in range(12):
        layer_entropies = []
        layer_erfs = []
        layer_ginis = []
        for att_batch in all_attentions:
            attn = att_batch[layer_idx]  # (B, H, 50, 50)
            cls_attn = attn[:, :, 0, 1:]  # (B, H, 49)
            cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-10)
            
            from src.metrics import attention_entropy, erf_at_threshold, gini_coefficient
            ent = attention_entropy(cls_attn)  # (B, H)
            erf = erf_at_threshold(cls_attn, 0.95)  # (B, H)
            
            B, H, N = cls_attn.shape
            gini = gini_coefficient(cls_attn.reshape(B*H, N)).reshape(B, H)
            
            layer_entropies.append(ent.cpu().numpy())
            layer_erfs.append(erf.cpu().numpy())
            layer_ginis.append(gini.cpu().numpy())
        
        # Stack: (total_images, H)
        layer_ent = np.concatenate(layer_entropies, axis=0)
        layer_erf = np.concatenate(layer_erfs, axis=0)
        layer_gini = np.concatenate(layer_ginis, axis=0)
        
        per_layer_per_head['entropy'].append({
            'mean_per_head': layer_ent.mean(axis=0).tolist(),
            'std_per_head': layer_ent.std(axis=0).tolist(),
            'mean': float(layer_ent.mean()),
            'std': float(layer_ent.std())
        })
        per_layer_per_head['erf95'].append({
            'mean_per_head': layer_erf.mean(axis=0).tolist(),
            'std_per_head': layer_erf.std(axis=0).tolist(),
            'mean': float(layer_erf.mean()),
            'std': float(layer_erf.std())
        })
        per_layer_per_head['gini'].append({
            'mean_per_head': layer_gini.mean(axis=0).tolist(),
            'std_per_head': layer_gini.std(axis=0).tolist(),
            'mean': float(layer_gini.mean()),
            'std': float(layer_gini.std())
        })
        
        # TensorBoard
        writer.add_scalar(f'baseline/entropy_layer_{layer_idx}', layer_ent.mean(), 0)
        writer.add_scalar(f'baseline/erf95_layer_{layer_idx}', layer_erf.mean(), 0)
        writer.add_scalar(f'baseline/gini_layer_{layer_idx}', layer_gini.mean(), 0)
    
    # Save baseline results
    baseline_result = {
        'summary': baseline_metrics,
        'per_layer_per_head': per_layer_per_head,
        'model': 'openai/clip-vit-base-patch32',
        'eval_set_size': 200,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    result_path = OUTPUT_DIR / "metrics" / "E1_baseline_stats.json"
    with open(result_path, 'w') as f:
        json.dump(baseline_result, f, indent=2)
    
    writer.close()
    
    print(f"\nBaseline Results:")
    print(f"  Mean Entropy:        {baseline_metrics['entropy_mean']:.4f}")
    print(f"  Mean ERF@0.95:       {baseline_metrics['erf95_mean']:.4f}")
    print(f"  Mean Gini:           {baseline_metrics['gini_mean']:.4f}")
    print(f"  Mean Head Diversity: {baseline_metrics['head_diversity_mean']:.4f}")
    print(f"\n  Per-layer Entropy: {[f'{e:.3f}' for e in baseline_metrics['entropy_per_layer']]}")
    print(f"  Per-layer ERF@0.95: {[f'{e:.3f}' for e in baseline_metrics['erf95_per_layer']]}")
    
    del model
    torch.cuda.empty_cache()
    
    return baseline_result


def run_full_ft_experiment(dataset_name, lr=1e-5, num_epochs=20, experiment_id=None,
                            weight_decay=0.01, freeze_layers=0, baseline_metrics=None):
    """Run Full Fine-tuning experiment."""
    set_seed(SEED)
    
    if dataset_name == "eurosat":
        train_dataset, test_dataset, num_classes, class_names = load_eurosat(cache_dir="./data")
    else:
        train_dataset, test_dataset, num_classes, class_names = load_oxford_pets(cache_dir="./data")
    
    if experiment_id is None:
        experiment_id = f"full_ft_{dataset_name}_lr{lr}"
    
    model = CLIPClassifier(num_classes=num_classes).to(DEVICE)
    
    # Optionally freeze layers
    if freeze_layers > 0:
        for i in range(freeze_layers):
            for p in model.vision_model.encoder.layers[i].parameters():
                p.requires_grad = False
    
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    eval_subset, _ = create_fixed_eval_subset(test_dataset, num_samples=200,
                                                num_classes=num_classes, seed=SEED)
    eval_loader = get_dataloader(eval_subset, batch_size=32, shuffle=False, num_workers=2)
    
    tb_dir = OUTPUT_DIR / "logs" / "tensorboard" / experiment_id
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    config = {
        'method': 'full_ft', 'dataset': dataset_name, 'lr': lr,
        'num_epochs': num_epochs, 'batch_size': 64, 'weight_decay': weight_decay,
        'freeze_layers': freeze_layers, 'model_name': 'openai/clip-vit-base-patch32'
    }
    
    history = train_one_experiment(config, model, train_loader, val_loader, 
                                   eval_loader, writer, experiment_id)
    writer.close()
    
    del model
    torch.cuda.empty_cache()
    return history


def run_lora_experiment(dataset_name, lora_r=8, lr=1e-4, num_epochs=20,
                         experiment_id=None, target_modules=None):
    """Run LoRA Fine-tuning experiment."""
    set_seed(SEED)
    
    if dataset_name == "eurosat":
        train_dataset, test_dataset, num_classes, class_names = load_eurosat(cache_dir="./data")
    else:
        train_dataset, test_dataset, num_classes, class_names = load_oxford_pets(cache_dir="./data")
    
    if experiment_id is None:
        experiment_id = f"lora_r{lora_r}_{dataset_name}_lr{lr}"
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    model = create_lora_model(
        num_classes=num_classes, lora_r=lora_r,
        lora_alpha=2*lora_r, target_modules=target_modules
    )
    
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    eval_subset, _ = create_fixed_eval_subset(test_dataset, num_samples=200,
                                                num_classes=num_classes, seed=SEED)
    eval_loader = get_dataloader(eval_subset, batch_size=32, shuffle=False, num_workers=2)
    
    tb_dir = OUTPUT_DIR / "logs" / "tensorboard" / experiment_id
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    tm_str = "_".join(target_modules)
    config = {
        'method': 'lora', 'dataset': dataset_name, 'lr': lr,
        'num_epochs': num_epochs, 'batch_size': 64, 'weight_decay': 0.01,
        'lora_r': lora_r, 'lora_alpha': 2*lora_r, 'target_modules': target_modules,
        'model_name': 'openai/clip-vit-base-patch32'
    }
    
    history = train_one_experiment(config, model, train_loader, val_loader,
                                   eval_loader, writer, experiment_id)
    writer.close()
    
    del model
    torch.cuda.empty_cache()
    return history


def run_regularization_experiment(reg_type, lambda_reg, baseline_metrics, 
                                    dataset_name="eurosat", lr=1e-5, num_epochs=20):
    """Run regularization experiment (APR or Entropy Floor)."""
    set_seed(SEED)
    
    if dataset_name == "eurosat":
        train_dataset, test_dataset, num_classes, class_names = load_eurosat(cache_dir="./data")
    else:
        train_dataset, test_dataset, num_classes, class_names = load_oxford_pets(cache_dir="./data")
    
    experiment_id = f"reg_{reg_type}_lambda{lambda_reg}_{dataset_name}"
    
    model = CLIPClassifier(num_classes=num_classes).to(DEVICE)
    
    # Create regularizer
    if reg_type == "apr":
        # Load a fresh pretrained vision model for APR
        from transformers import CLIPModel
        pretrained = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", attn_implementation="eager"
        ).vision_model
        regularizer = AttentionPreservationRegularizer(
            pretrained, lambda_reg=lambda_reg, device=DEVICE
        )
    elif reg_type == "entropy_floor":
        regularizer = EntropyFloorRegularizer(
            baseline_entropy_per_layer=baseline_metrics['entropy_per_layer'],
            floor_ratio=0.7, lambda_reg=lambda_reg
        )
    else:
        regularizer = None
    
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    eval_subset, _ = create_fixed_eval_subset(test_dataset, num_samples=200,
                                                num_classes=num_classes, seed=SEED)
    eval_loader = get_dataloader(eval_subset, batch_size=32, shuffle=False, num_workers=2)
    
    tb_dir = OUTPUT_DIR / "logs" / "tensorboard" / experiment_id
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    config = {
        'method': f'full_ft+{reg_type}', 'dataset': dataset_name, 'lr': lr,
        'num_epochs': num_epochs, 'batch_size': 64, 'weight_decay': 0.01,
        'reg_type': reg_type, 'lambda_reg': lambda_reg,
        'model_name': 'openai/clip-vit-base-patch32'
    }
    
    history = train_one_experiment(config, model, train_loader, val_loader,
                                   eval_loader, writer, experiment_id,
                                   regularizer=regularizer)
    writer.close()
    
    # Clean up APR pretrained model
    if reg_type == "apr":
        del regularizer.pretrained_model
    del model
    torch.cuda.empty_cache()
    return history


def run_zero_shot_evaluation(model_path, config_override=None):
    """Evaluate a checkpoint on zero-shot tasks using CLIP text encoder."""
    # For zero-shot, we use CLIP's text-image matching
    from transformers import CLIPModel, CLIPProcessor
    
    print("  Running zero-shot evaluation...")
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", attn_implementation="eager"
    ).to(DEVICE)
    
    # If given a checkpoint, load vision model weights
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        # Only load vision model weights (may be partial)
        state = checkpoint.get('model_state_dict', checkpoint)
        vision_state = {k.replace('vision_model.', ''): v for k, v in state.items() 
                       if k.startswith('vision_model.')}
        if vision_state:
            clip_model.vision_model.load_state_dict(vision_state, strict=False)
    
    clip_model.eval()
    results = {}
    
    # Evaluate on CIFAR-100
    try:
        cifar_test, cifar_classes, cifar_names = load_cifar100(cache_dir="./data")
        cifar_loader = get_dataloader(cifar_test, batch_size=64, shuffle=False, num_workers=2)
        
        # Create text prompts
        text_inputs = processor(
            text=[f"a photo of a {name}" for name in cifar_names],
            return_tensors="pt", padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        correct = 0
        total = 0
        for images, labels in cifar_loader:
            images = images.to(DEVICE)
            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T)
                predicted = similarity.argmax(dim=-1)
            
            labels = labels.to(DEVICE)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        results['cifar100'] = correct / total
        print(f"    CIFAR-100 zero-shot: {results['cifar100']:.4f}")
    except Exception as e:
        print(f"    CIFAR-100 evaluation failed: {e}")
        results['cifar100'] = None
    
    del clip_model
    torch.cuda.empty_cache()
    return results


# =====================================================
# MAIN EXECUTION
# =====================================================
def main():
    set_seed(SEED)
    os.chdir("/workspace/project")
    
    start_time = time.time()
    all_results = {}
    
    # =====================================================
    # PHASE 1: Baseline Analysis
    # =====================================================
    print("\n" + "#"*70)
    print("# PHASE 1: BASELINE ANALYSIS")
    print("#"*70)
    
    baseline = run_baseline_analysis()
    all_results['E1_baseline'] = baseline
    
    # =====================================================
    # PHASE 2: Main Experiments
    # =====================================================
    print("\n" + "#"*70)
    print("# PHASE 2: MAIN EXPERIMENTS")
    print("#"*70)
    
    # E2: Full FT on EuroSAT
    print("\n--- E2: Full FT EuroSAT ---")
    e2 = run_full_ft_experiment("eurosat", lr=1e-5, num_epochs=20, 
                                 experiment_id="E2_full_ft_eurosat")
    all_results['E2_full_ft_eurosat'] = e2
    
    # E3: Full FT on Oxford Pets
    print("\n--- E3: Full FT Oxford Pets ---")
    e3 = run_full_ft_experiment("pets", lr=1e-5, num_epochs=30, 
                                 experiment_id="E3_full_ft_pets")
    all_results['E3_full_ft_pets'] = e3
    
    # E4: LoRA r=4 on EuroSAT
    print("\n--- E4: LoRA r=4 EuroSAT ---")
    e4 = run_lora_experiment("eurosat", lora_r=4, lr=1e-4, num_epochs=20,
                              experiment_id="E4_lora_r4_eurosat")
    all_results['E4_lora_r4_eurosat'] = e4
    
    # E5: LoRA r=8 on EuroSAT
    print("\n--- E5: LoRA r=8 EuroSAT ---")
    e5 = run_lora_experiment("eurosat", lora_r=8, lr=1e-4, num_epochs=20,
                              experiment_id="E5_lora_r8_eurosat")
    all_results['E5_lora_r8_eurosat'] = e5
    
    # E6: LoRA r=16 on EuroSAT
    print("\n--- E6: LoRA r=16 EuroSAT ---")
    e6 = run_lora_experiment("eurosat", lora_r=16, lr=1e-4, num_epochs=20,
                              experiment_id="E6_lora_r16_eurosat")
    all_results['E6_lora_r16_eurosat'] = e6
    
    # E7: LoRA r=8 on Oxford Pets
    print("\n--- E7: LoRA r=8 Oxford Pets ---")
    e7 = run_lora_experiment("pets", lora_r=8, lr=1e-4, num_epochs=30,
                              experiment_id="E7_lora_r8_pets")
    all_results['E7_lora_r8_pets'] = e7
    
    # =====================================================
    # PHASE 3: Ablation Experiments
    # =====================================================
    print("\n" + "#"*70)
    print("# PHASE 3: ABLATION EXPERIMENTS")
    print("#"*70)
    
    # A1: Learning rate sweep
    print("\n--- A1: Learning Rate Ablation ---")
    for lr in [1e-6, 5e-6, 5e-5, 1e-4]:
        exp_id = f"A1_lr_{lr}"
        print(f"\n  LR = {lr}")
        a1 = run_full_ft_experiment("eurosat", lr=lr, num_epochs=20,
                                     experiment_id=exp_id)
        all_results[exp_id] = a1
    
    # A2: LoRA target modules
    print("\n--- A2: LoRA Target Modules Ablation ---")
    a2_qkvo = run_lora_experiment("eurosat", lora_r=8, lr=1e-4, num_epochs=20,
                                   experiment_id="A2_lora_qkvo",
                                   target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
    all_results['A2_lora_qkvo'] = a2_qkvo
    
    # A3: Frozen layers
    print("\n--- A3: Frozen Layers Ablation ---")
    for freeze_n in [3, 6, 9]:
        exp_id = f"A3_freeze_{freeze_n}"
        print(f"\n  Freeze first {freeze_n} layers")
        a3 = run_full_ft_experiment("eurosat", lr=1e-5, num_epochs=20,
                                     experiment_id=exp_id, freeze_layers=freeze_n)
        all_results[exp_id] = a3
    
    # =====================================================
    # PHASE 4: Regularization Experiments
    # =====================================================
    print("\n" + "#"*70)
    print("# PHASE 4: REGULARIZATION EXPERIMENTS")
    print("#"*70)
    
    baseline_summary = baseline['summary']
    
    # R1: APR (KL divergence)
    print("\n--- R1: APR Regularization ---")
    for lam in [0.01, 0.1, 1.0]:
        exp_id = f"R1_apr_lambda{lam}"
        print(f"\n  APR λ = {lam}")
        r1 = run_regularization_experiment("apr", lam, baseline_summary,
                                            num_epochs=20)
        all_results[exp_id] = r1
    
    # R2: Entropy Floor
    print("\n--- R2: Entropy Floor Regularization ---")
    for lam in [0.01, 0.1]:
        exp_id = f"R2_entropy_floor_lambda{lam}"
        print(f"\n  Entropy Floor λ = {lam}")
        r2 = run_regularization_experiment("entropy_floor", lam, baseline_summary,
                                            num_epochs=20)
        all_results[exp_id] = r2
    
    # R3: Weight Decay comparison
    print("\n--- R3: Weight Decay Ablation ---")
    for wd in [0.0, 0.1]:
        exp_id = f"R3_wd_{wd}"
        print(f"\n  Weight Decay = {wd}")
        r3 = run_full_ft_experiment("eurosat", lr=1e-5, num_epochs=20,
                                     experiment_id=exp_id, weight_decay=wd)
        all_results[exp_id] = r3
    
    # =====================================================
    # PHASE 5: Zero-shot Evaluation
    # =====================================================
    print("\n" + "#"*70)
    print("# PHASE 5: ZERO-SHOT EVALUATION")
    print("#"*70)
    
    # Evaluate pretrained (baseline)
    zs_baseline = run_zero_shot_evaluation(None)
    all_results['zero_shot_baseline'] = zs_baseline
    
    # Evaluate key checkpoints
    for exp_id in ['E2_full_ft_eurosat', 'E3_full_ft_pets', 'E5_lora_r8_eurosat']:
        ckpt_path = OUTPUT_DIR / "checkpoints" / exp_id / "best_model.pth"
        if ckpt_path.exists():
            print(f"\n  Zero-shot eval for {exp_id}")
            zs = run_zero_shot_evaluation(str(ckpt_path))
            all_results[f'zero_shot_{exp_id}'] = zs
    
    # =====================================================
    # Save all results
    # =====================================================
    total_time = time.time() - start_time
    
    summary = {
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'num_experiments': len(all_results),
        'timestamp': datetime.datetime.now().isoformat(),
        'device': str(DEVICE),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    }
    
    # Save master results
    master_path = OUTPUT_DIR / "metrics" / "all_results_summary.json"
    with open(master_path, 'w') as f:
        json.dump({'summary': summary, 'experiments': {k: str(type(v)) for k, v in all_results.items()}}, 
                  f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {OUTPUT_DIR}/metrics/")
    print(f"TensorBoard logs: {OUTPUT_DIR}/logs/tensorboard/")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    results = main()
