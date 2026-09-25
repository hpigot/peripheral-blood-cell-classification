import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bloodcell.model import INPUT_SIZE, build_model, transforms  # noqa: E402


@pytest.mark.parametrize("arch", ["mobilenet_v3_small", "resnet18"])
def test_forward_shape(arch):
    m = build_model(arch, num_classes=8, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE))
    assert out.shape == (2, 8)


def test_eval_transform_output_shape():
    from PIL import Image

    x = transforms(train=False)(Image.new("RGB", (360, 363)))
    assert tuple(x.shape) == (3, INPUT_SIZE, INPUT_SIZE)
