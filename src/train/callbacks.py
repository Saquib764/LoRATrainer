import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os

try:
    import wandb
except ImportError:
    wandb = None

# from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["model_type"][
                    0
                ],
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        model_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        target_size = trainer.training_config["dataset"]["target_size"]

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        if model_type == "cartoon":
            test_list.append("A cartoon character in a white background. He is looking right, and running.")
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, prompt in enumerate(test_list):
            res = pl_module.flux_pipe(
                prompt=prompt,
                height=target_size,
                width=target_size,
                generator=generator
            )
            # res = generate(
            #     pl_module.flux_pipe,
            #     prompt=prompt,
            #     height=target_size,
            #     width=target_size,
            #     generator=generator,
            #     model_config=pl_module.model_config,
            #     default_lora=True,
            # )
            res.images[0].save(
                os.path.join(save_path, f"{file_name}_{i}.jpg")
            )