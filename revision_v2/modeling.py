from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _require_transformers():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The revision pipeline requires `transformers`. "
            "Install it before running model or training commands."
        ) from exc
    return CLIPModel, CLIPProcessor


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The revision pipeline requires `peft` for LoRA runs. "
            "Install it before running LoRA experiments."
        ) from exc
    return LoraConfig, get_peft_model


class RevisionCLIPClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int):
        super().__init__()
        CLIPModel, _ = _require_transformers()
        self.clip_model = CLIPModel.from_pretrained(backbone, attn_implementation="eager")
        self.backbone = backbone
        projection_dim = self.clip_model.config.projection_dim
        self.classifier = nn.Linear(projection_dim, num_classes)

        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        text_projection = getattr(self.clip_model, "text_projection", None)
        if isinstance(text_projection, nn.Module):
            for param in text_projection.parameters():
                param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor, *, output_attentions: bool = False, output_hidden_states: bool = False):
        outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        pooled = outputs.pooler_output
        image_embeds = self.clip_model.visual_projection(pooled)
        return outputs, image_embeds

    def forward(self, pixel_values: torch.Tensor, *, output_attentions: bool = False, output_hidden_states: bool = False):
        vision_outputs, image_embeds = self.encode_image(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.classifier(image_embeds)
        return {
            "logits": logits,
            "attentions": vision_outputs.attentions if output_attentions else None,
            "hidden_states": vision_outputs.hidden_states if output_hidden_states else None,
            "image_embeds": image_embeds,
        }

    def get_zero_shot_logits(self, pixel_values: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        _, image_embeds = self.encode_image(pixel_values, output_attentions=False, output_hidden_states=False)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        return logit_scale * image_embeds @ text_embeds.T


@dataclass
class BuiltModel:
    model: RevisionCLIPClassifier
    processor: object


def build_model(
    backbone: str,
    num_classes: int,
    method: str,
    *,
    lora_targets: list[str] | None = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> BuiltModel:
    _, CLIPProcessor = _require_transformers()
    model = RevisionCLIPClassifier(backbone=backbone, num_classes=num_classes)
    processor = CLIPProcessor.from_pretrained(backbone)

    if method == "pretrained":
        for param in model.parameters():
            param.requires_grad = False
        return BuiltModel(model=model, processor=processor)

    if method == "full_ft":
        for param in model.clip_model.vision_model.parameters():
            param.requires_grad = True
        for param in model.clip_model.visual_projection.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        return BuiltModel(model=model, processor=processor)

    if method == "lora":
        LoraConfig, get_peft_model = _require_peft()
        lora_targets = lora_targets or ["q_proj", "v_proj"]
        for param in model.parameters():
            param.requires_grad = False
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_targets,
            bias="none",
        )
        model.clip_model.vision_model = get_peft_model(model.clip_model.vision_model, config)
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.clip_model.visual_projection.parameters():
            param.requires_grad = True
        return BuiltModel(model=model, processor=processor)

    raise ValueError(f"Unsupported method: {method}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
