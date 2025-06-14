from transformers import Trainer
import torch
import torch.nn.functional as F


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Example: weighted cross-entropy
        weights = [1.0]*20
        class_weights = torch.tensor(weights, device=logits.device)  # Customize this
        loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="sum")

        # Normalize using provided batch size (not inferred from labels)
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        return (loss, outputs) if return_outputs else loss