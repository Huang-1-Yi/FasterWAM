from copy import deepcopy

import torch

from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor


class CachedActionProcessor(FastWAMProcessor):
    """FastWAM processor for cached-latent training; intentionally skips images."""

    def preprocess(self, data):
        sample = {
            "instruction": self.augment_instruction(data),
            "image_is_pad": data["image_is_pad"],
        }

        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.delta_action_dim_mask.items():
                    current_action = data["action"][key]
                    current_pad = action_is_pad.to(device=current_action.device)
                    current_dim_mask = dim_mask.to(device=current_action.device)
                    current_action[current_pad.unsqueeze(1) & current_dim_mask.unsqueeze(0)] = 0.0

        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"]
            sample["action_is_pad"] = data["action_is_pad"]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]
            if int(sample["action"].shape[-1]) != int(self.action_output_dim):
                raise ValueError("Processed action dimension mismatch.")

        sample["proprio"] = data["state"]
        sample["proprio_is_pad"] = data["state_is_pad"]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"]
        if int(sample["proprio"].shape[-1]) != int(self.proprio_output_dim):
            raise ValueError("Processed proprio dimension mismatch.")
        sample["idx"] = data["idx"]
        return sample
