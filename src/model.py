"""
model.py - CLIP Classifier and model utilities
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model


class CLIPClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_classes=10):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(
            model_name, attn_implementation="eager"
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
    model = CLIPModel.from_pretrained(model_name, attn_implementation="eager")
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
