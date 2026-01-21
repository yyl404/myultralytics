"""Calculate parameter importance for YOLO models using Fisher Information Matrix.

This script calculates the importance of each parameter in a YOLO model based on
the Fisher Information Matrix (diagonal approximation), which is computed as the
square of gradients averaged over the training dataset.

Usage:
    $ python tools/cal_importance.py \
        --model <path/to/model.pt> \
        --dataset <path/to/dataset.yaml> \
        --save_path <path/to/save/importance.pth> \
        [--batch_size <batch_size>] \
        [--workers <num_workers>] \
        [--device <device>] \
        [--layers <layer1> <layer2> ...] \
        [--modules <module1> <module2> ...]

Arguments:
    --model: Path to the model checkpoint (.pt file)
    --dataset: Path to the dataset YAML file
    --save_path: Path to save the importance dictionary
    --batch_size: Batch size for processing (default: 16)
    --workers: Number of data loading workers (default: 8)
    --device: Device to use (default: "cuda")
    --layers: (optional) Layers to calculate importance for, space-separated.
        If specified, only parameters in these layers will be calculated.
    --modules: (optional) Specific module names to calculate importance for, space-separated.
        Provides more detailed control than --layers. Module names should match
        the parameter name prefix (e.g., "model.10.conv", "model.11.bn").

Example:
    $ python tools/cal_importance.py \
        --model runs/task1/best.pt \
        --dataset data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
        --save_path runs/task1/importance.pth \
        --batch_size 16 \
        --device 0
    
    $ python tools/cal_importance.py \
        --model runs/task1/best.pt \
        --dataset data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
        --save_path runs/task1/importance.pth \
        --layers 10 11 12
    
    $ python tools/cal_importance.py \
        --model runs/task1/best.pt \
        --dataset data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
        --save_path runs/task1/importance.pth \
        --modules model.10.conv model.11.bn
"""

import argparse
import os
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.callbacks import default_callbacks
from ultralytics.models.yolo.detect import DetectionTrainer


class ImportanceCalculator:
    def __init__(self, model, layers=None, modules=None, device="cuda"):
        """Initialize ImportanceCalculator.
        
        Args:
            model: Model to calculate importance for
            layers: List of layer IDs to calculate importance for (optional)
            modules: List of module name prefixes to calculate importance for (optional)
            device: Device to use (default: "cuda")
        """
        self.model = model
        self.modules = {}
        self.running_importance = {}
        self.n_batch = {}
        if not torch.cuda.is_available():
            device = "cpu"
            LOGGER.warning("CUDA is not available, using CPU")
        self.device = device
        
        def _match(n, lid):
            return f"model.{lid}." in n and "dfl" not in n

        if modules is not None:
            for n, m in model.named_modules():
                if n in modules:
                    self.modules[n] = m
        elif layers is not None:
            for lid in layers:
                for n, m in model.named_modules():
                    if _match(n, lid):
                        self.modules[n] = m
        else:
            for n, m in model.named_modules():
                if "dfl" not in n:
                    self.modules[n] = m

    @property
    def names(self):
        return list(self.running_importance.keys())

    def process_gradients(self):
        """Process gradients for all tracked modules after backward() completes.
        This should be called after loss.backward() but before optimizer.step().
        
        This method applies gradient clipping to prevent NaN/Inf values before processing.
        """
        max_grad_norm = 35.0  # Clip gradients to prevent overflow when squared
        
        for module_name, mod in self.modules.items():
            for param_name, param in mod.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                    
                    # Apply gradient clipping: replace NaN/Inf and clip to reasonable range
                    param.grad = torch.nan_to_num(
                        param.grad, 
                        nan=0.0, 
                        posinf=max_grad_norm, 
                        neginf=-max_grad_norm
                    )
                    param.grad = torch.clamp(param.grad, min=-max_grad_norm, max=max_grad_norm)
                    
                    # Calculate gradient squared (now safe from overflow)
                    grad_squared = param.grad ** 2
                    
                    # Normalize gradient squared
                    grad_max_val = torch.max(grad_squared).item()
                    grad_squared_normalized = grad_squared / (grad_max_val + 1e-12)
                    
                    if full_param_name not in self.running_importance.keys():
                        self.running_importance[full_param_name] = torch.zeros_like(grad_squared_normalized, device=param.device)
                        self.n_batch[full_param_name] = 0
                    
                    n_batch = self.n_batch[full_param_name]
                    self.running_importance[full_param_name] = n_batch / (n_batch + 1) * self.running_importance[full_param_name] + \
                        grad_squared_normalized / (n_batch+1)
                    self.n_batch[full_param_name] += 1

    def save_importance(self, save_path):
        """Save calculator state to file.
        
        Saves all state variables needed to restore the calculator:
        - running_importance: Dictionary of parameter importance values
        - n_batch: Dictionary of batch counts for each parameter
        - modules: List of module names (keys of self.modules)
        - device: Device used for calculation
        """
        LOGGER.info(f"Saving importance to {save_path}")
        state = {
            'running_importance': self.running_importance,
            'n_batch': self.n_batch,
            'modules': list(self.modules.keys()),
            'device': self.device,
        }
        with open(save_path, "wb") as f:
            torch.save(state, f)
    
    def load_importance(self, load_path):
        """Load calculator state from file.
        
        Restores all state variables:
        - running_importance: Dictionary of parameter importance values
        - n_batch: Dictionary of batch counts for each parameter
        - modules: Dictionary of modules restored from saved module names
        - device: Device used for calculation
        
        Note: Requires self.model to be set before calling this method to restore modules.
        """
        with open(load_path, "rb") as f:
            state = torch.load(f)
        
        self.running_importance = state.get('running_importance', {})
        self.n_batch = state.get('n_batch', {})
        self.device = state.get('device', self.device)
        
        module_names = state.get('modules', None)
        if module_names is not None and self.model is not None:
            self.modules = {}
            for n, m in self.model.named_modules():
                if n in module_names:
                    self.modules[n] = m
        
        LOGGER.info(f"Loaded importance from {load_path}")


def calculate_importance(model, dataset, layers=None, modules=None, 
                         workers=8, device="cuda", epochs=1, batch_size=None, load_hist=None):
    """Calculate parameter importance using Fisher Information Matrix approximation.
    
    Args:
        model: YOLO model instance
        dataset: Path to dataset YAML file or dataset config dict
        layers: List of layer IDs to calculate importance for (optional)
        modules: List of module name prefixes to calculate importance for (optional)
            If both layers and modules are None, importance will be calculated for all modules
            (excluding DFL layer which is always frozen)
        workers: Number of data loading workers
        device: Device to use for training
        epochs: Number of epochs to train (default: 1)
        batch_size: Batch size for training (optional, passed to model.train() if specified)
        load_hist: Path to previously saved importance file to load as starting point (optional)
    
    Returns:
        ImportanceCalculator: Calculator instance with calculated importance
    """
    calculator = None
    
    train_kwargs = {
        'data': dataset,
        'epochs': epochs,
        'device': device,
        'workers': workers,
        'batch': batch_size,
        'val': False,
        'plots': False,
        'save': False,
        'amp': False, # Use float32 to achieve high accuracy
        'model': model.ckpt_path if hasattr(model, 'ckpt_path') else str(model.overrides.get('model', '')),
    }
    
    def on_train_start_callback(trainer):
        """Callback function executed at the start of training.
        
        This callback initializes the importance calculator and overrides the optimizer
        step to collect gradient statistics for importance calculation without actually
        updating model parameters.
        """
        nonlocal calculator
        
        # Initialize importance calculator for tracking parameter importance
        calculator = ImportanceCalculator(
            trainer.model,
            layers=layers,
            modules=modules,
            device=device
        )
        
        # Load previously calculated importance as starting point (if provided)
        # This allows incremental importance calculation across multiple training runs
        if load_hist is not None:
            calculator.load_importance(load_hist)
            LOGGER.info(f"Loaded previous importance from {load_hist} as starting point")
        
        # Note: Model parameters' requires_grad is managed by the trainer
        # for param in trainer.model.parameters():
        #     param.requires_grad = True
        
        LOGGER.info(f"Registered importance calculation for {len(calculator.modules)} modules")

        def new_optimizer_step():
            """Override optimizer step to collect gradients for importance calculation without updating model."""
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
            calculator.process_gradients()
            # Skip optimizer.step() - only collect statistics, don't update model
            # trainer.scaler.step(trainer.optimizer)
            # trainer.scaler.update()
            trainer.optimizer.zero_grad()
            # if trainer.ema:
            #     trainer.ema.update(trainer.model)
        
        # Override the default optimizer step with our custom version
        trainer.optimizer_step = new_optimizer_step
        
    # Register the callback to be executed when training starts
    callbacks = default_callbacks.copy()
    callbacks['on_train_start'] = callbacks['on_train_start'] + [on_train_start_callback]
    
    # Disable validation during and after training
    def disable_validation(trainer):
        # Ensure val is False
        trainer.args.val = False
        
        # Override validate method to skip validation
        original_validate = trainer.validate
        def no_op_validate():
            LOGGER.debug("Skipping validation for importance calculation")
            return {}, 0.0  # Return empty metrics and zero fitness
        trainer.validate = no_op_validate
        
        # Override final_eval method to skip final evaluation
        original_final_eval = trainer.final_eval
        def no_op_final_eval():
            LOGGER.info("Skipping final evaluation for importance calculation")
            pass
        trainer.final_eval = no_op_final_eval
    
    callbacks['on_train_start'] = callbacks['on_train_start'] + [disable_validation]
    
    trainer = DetectionTrainer(overrides=train_kwargs, _callbacks=callbacks)
    
    trainer.train()
    
    LOGGER.info("Importance calculation completed!")
    return calculator


def main():
    parser = argparse.ArgumentParser(description="Calculate parameter importance for YOLO models")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to the dataset YAML file")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Path to save the importance dictionary")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training (optional, will use model default if not specified)")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of data loading workers (default: 8)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                       help="Layers to calculate importance for, space-separated. "
                            "If specified, only parameters in these layers will be calculated. "
                            "If both --layers and --modules are not specified, importance will be "
                            "calculated for all modules (excluding DFL layer).")
    parser.add_argument("--modules", nargs="+", type=str, default=None,
                       help="Specific module names to calculate importance for, space-separated. "
                            "Provides more detailed control than --layers. Module names should "
                            "match the parameter name prefix (e.g., 'model.10.conv', 'model.11.bn'). "
                            "If both --layers and --modules are not specified, importance will be "
                            "calculated for all modules (excluding DFL layer).")
    parser.add_argument("--load_hist", type=str, default=None,
                       help="Path to previously saved importance file to load as starting point. "
                            "If specified, importance calculation will continue from the loaded state.")
    
    args = parser.parse_args()
    
    LOGGER.info(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    importance_calculator = calculate_importance(
        model, 
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        layers=args.layers,
        modules=args.modules,
        load_hist=args.load_hist
    )
    
    LOGGER.info(f"Saving importance to {args.save_path}...")
    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    importance_calculator.save_importance(args.save_path)
    
    total_params = sum(p.numel() for p in importance_calculator.running_importance.values())
    LOGGER.info(f"Importance calculation completed!")
    LOGGER.info(f"Total parameters with importance: {len(importance_calculator.running_importance)}")
    LOGGER.info(f"Total parameter count: {total_params:,}")
    LOGGER.info(f"Importance saved to: {args.save_path}")


if __name__ == "__main__":
    main()

