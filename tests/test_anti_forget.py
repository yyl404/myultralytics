"""Tests for anti-forget teacher decode-path sync."""

from types import SimpleNamespace

from ultralytics.engine.anti_forget import _apply_train_head_args


class _DummyDetectModel:
    def __init__(self, end2end=True):
        self.end2end = end2end
        self.head_attrs = {}

    def set_head_attr(self, **kwargs):
        self.head_attrs.update(kwargs)


def test_apply_train_head_args_forces_one2many_nms_on_teacher():
    model = _DummyDetectModel(end2end=True)
    args = SimpleNamespace(end2end=False, agnostic_nms=True, max_det=300)
    _apply_train_head_args(model, args)
    assert model.end2end is False
    assert model.head_attrs["agnostic_nms"] is True
    assert "max_det" not in model.head_attrs


def test_apply_train_head_args_keeps_end2end_max_det():
    model = _DummyDetectModel(end2end=True)
    args = SimpleNamespace(end2end=True, agnostic_nms=False, max_det=100)
    _apply_train_head_args(model, args)
    assert model.end2end is True
    assert model.head_attrs["max_det"] == 100
    assert model.head_attrs["agnostic_nms"] is False
