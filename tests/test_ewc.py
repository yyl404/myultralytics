"""Focused tests for paper-aligned EWC state, Fisher accumulation, and head expansion."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from tools.cal_importance import ImportanceCalculator
from tools.expand_importance import expand_ewc_state
from ultralytics.engine.ewc import EWC_STATE_VERSION, EWCLoss, validate_ewc_state


def _state(importance, task_params, **metadata):
    return {
        "version": EWC_STATE_VERSION,
        "importance": importance,
        "task_params": task_params,
        **metadata,
    }


class _Head(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.cv3 = nn.ModuleList(
            [nn.Sequential(nn.Identity(), nn.Identity(), nn.Conv2d(1, classes, kernel_size=1))]
        )


class EWCTest(unittest.TestCase):
    def test_loss_sums_all_tasks_with_half_factor(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.copy_(torch.tensor([[2.0, -1.0]]))
        state = _state(
            importance={"weight": [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]},
            task_params={"weight": [torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0, -2.0]])]},
        )

        loss = EWCLoss(model=model, state=state).get_loss()

        expected = 0.5 * (1.0 * 1.0**2 + 2.0 * (-2.0) ** 2 + 3.0 * 2.0**2 + 4.0 * 1.0**2)
        self.assertAlmostEqual(loss.item(), expected)
        loss.backward()
        self.assertIsNotNone(model.weight.grad)

    def test_state_validation_fails_on_inconsistent_history(self):
        state = _state(
            importance={"weight": [torch.ones(2)]},
            task_params={"weight": [torch.ones(2), torch.ones(2)]},
        )
        with self.assertRaisesRegex(ValueError, "history length"):
            validate_ewc_state(state)

    def test_importance_weights_batches_and_appends_history(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        calculator = ImportanceCalculator(model=model, scope="normalization")
        self.assertEqual(set(calculator.parameters), {"1.bias", "1.weight"})

        for parameter in calculator.parameters.values():
            parameter.grad = torch.ones_like(parameter)
        calculator.process_gradients(batch_samples=3)
        for parameter in calculator.parameters.values():
            parameter.grad = torch.full_like(parameter, 2.0)
        calculator.process_gradients(batch_samples=1)

        state = calculator.build_state()
        for fisher_history in state["importance"].values():
            self.assertEqual(len(fisher_history), 1)
            self.assertTrue(torch.allclose(fisher_history[0], torch.full_like(fisher_history[0], 1.75)))
        self.assertEqual(state["sample_counts"], [4])

        appended = ImportanceCalculator(model=model, scope="normalization", history=state)
        for parameter in appended.parameters.values():
            parameter.grad = torch.full_like(parameter, 3.0)
        appended.process_gradients(batch_samples=2)
        appended_state = appended.build_state()
        self.assertTrue(all(len(history) == 2 for history in appended_state["importance"].values()))
        self.assertEqual(appended_state["sample_counts"], [4, 2])

    def test_expand_zeros_new_class_fisher_and_uses_new_snapshot(self):
        old_head = _Head(classes=2)
        new_head = _Head(classes=3)
        old_wrapper = SimpleNamespace(model=old_head, names={0: "dog", 1: "cat"})
        new_wrapper = SimpleNamespace(model=new_head, names={0: "bird", 1: "cat", 2: "dog"})
        name = "cv3.0.2.weight"
        new_parameter = dict(new_head.named_parameters())[name].detach().clone()
        old_fisher = torch.tensor([[[[2.0]]], [[[5.0]]]])
        old_snapshot = torch.tensor([[[[7.0]]], [[[11.0]]]])
        state = _state(
            importance={name: [old_fisher]},
            task_params={name: [old_snapshot]},
        )

        expanded = expand_ewc_state(state, old_wrapper, new_wrapper)
        fisher = expanded["importance"][name][0]
        snapshot = expanded["task_params"][name][0]

        self.assertEqual(fisher[:, 0, 0, 0].tolist(), [0.0, 5.0, 2.0])
        self.assertEqual(snapshot[1, 0, 0, 0].item(), 11.0)
        self.assertEqual(snapshot[2, 0, 0, 0].item(), 7.0)
        self.assertEqual(snapshot[0, 0, 0, 0].item(), new_parameter[0, 0, 0, 0].item())


if __name__ == "__main__":
    unittest.main()
