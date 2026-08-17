"""
model.py - CLIP Classifier and model utilities
"""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model


class CLIPClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_classes=10):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            use_safetensors=True,
        )
        self.vision_model = self.clip_model.vision_model
        self.visual_projection = self.clip_model.visual_projection  # projects to 512
        self.classifier = nn.Linear(512, num_classes)
        
        # Freeze text encoder
        for p in self.clip_model.text_model.parameters():
            p.requires_grad = False
        for p in self.clip_model.text_projection.parameters():
            p.requires_grad = False
    
    def forward(self, pixel_values, output_attentions=False):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions
        )
        pooled = vision_outputs.pooler_output  # CLS token
        projected = self.visual_projection(pooled)
        logits = self.classifier(projected)
        attentions = vision_outputs.attentions if output_attentions else None
        return logits, attentions
    
    def get_attention_maps(self, pixel_values):
        """Get attention maps without computing classification head."""
        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=True
            )
        return vision_outputs.attentions


def create_lora_model(model_name="openai/clip-vit-base-patch32", num_classes=10,
                      lora_r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=None):
    """Create a CLIP classifier with LoRA applied to vision encoder."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    base_model = CLIPClassifier(model_name, num_classes)
    
    # Freeze all parameters first
    for p in base_model.parameters():
        p.requires_grad = False
    
    # Apply LoRA to vision model
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    base_model.vision_model = get_peft_model(base_model.vision_model, lora_config)
    
    # Unfreeze classifier head
    for p in base_model.classifier.parameters():
        p.requires_grad = True
    for p in base_model.visual_projection.parameters():
        p.requires_grad = True
    
    return base_model


def get_pretrained_model(model_name="openai/clip-vit-base-patch32"):
    """Load pretrained CLIP model for attention analysis (no classification head)."""
    model = CLIPModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        use_safetensors=True,
    )
    model.eval()
    return model


def get_processor(model_name="openai/clip-vit-base-patch32"):
    """Get CLIP processor for image preprocessing."""
    return CLIPProcessor.from_pretrained(model_name)


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_classifier_from_checkpoint(checkpoint_path, map_location=None, eval_mode=True):
    """Rebuild a saved classifier or LoRA classifier from checkpoint config."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {})

    model_name = config.get("model_name", "openai/clip-vit-base-patch32")
    dataset_name = str(config.get("dataset", "")).lower()
    if "pet" in dataset_name:
        default_num_classes = 37
    elif "cifar100" in dataset_name:
        default_num_classes = 100
    else:
        default_num_classes = 10
    num_classes = config.get("num_classes", config.get("class_count", default_num_classes))
    method = str(config.get("method", "full_ft")).lower()

    if method.startswith("lora"):
        model = create_lora_model(
            model_name=model_name,
            num_classes=num_classes,
            lora_r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        )
    else:
        model = CLIPClassifier(model_name=model_name, num_classes=num_classes)

    model.load_state_dict(state, strict=False)
    if map_location is not None:
        model = model.to(map_location)
    if eval_mode:
        model.eval()
    return model, checkpoint, config
