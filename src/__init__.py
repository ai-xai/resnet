from .model import BasicBlock, ResNet


def get_resnet34(num_classes=2) -> ResNet:
    """Creates a ResNet-34 model.

    This configuration uses ``BasicBlock`` with layer depths
    ``[3, 4, 6, 3]``, matching the original ResNet-34 architecture.

    Args:
        num_classes (int, optional): Number of output classes for the
            final classification layer. Defaults to 2.

    Returns:
        ResNet: A ResNet-34 model instance.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
