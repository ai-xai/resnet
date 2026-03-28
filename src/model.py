import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """A standard residual block used in ResNet-18 and ResNet-34.

    This block consists of two 3x3 convolutional layers with batch
    normalization. A residual skip connection is added to the output.
    If the input and output dimensions differ, an optional downsampling
    layer is applied to the identity branch.

    Attributes:
        expansion (int): Multiplicative factor for output channels.
            For ``BasicBlock``, this is always 1.
    """

    expansion = 1

    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initializes the residual basic block.

        Args:
            C_in (int): Number of input channels.
            C_out (int): Number of output channels.
            stride (int, optional): Stride of the first convolutional
                layer. Defaults to 1.
            downsample (nn.Module | None, optional): Optional module used
                to match the residual branch dimensions with the main path.
                Defaults to None.
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch_size, C_in, H, W)``.

        Returns:
            torch.Tensor: Output tensor after residual addition and ReLU
            activation.
        """
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """A configurable ResNet architecture.

    This implementation supports ResNet variants based on residual
    blocks such as ResNet-18 and ResNet-34.

    The network consists of:
        - initial 7x7 convolution stem
        - max pooling
        - four residual stages
        - global average pooling
        - fully connected classification head

    Attributes:
        C_in (int): Current number of input channels used while
            constructing residual layers.
    """

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 2,
    ) -> None:
        """Initializes the ResNet model.

        Args:
            block (type[nn.Module]): Residual block class to use
                (e.g. ``BasicBlock``).
            layers (list[int]): Number of blocks in each of the four
                residual stages.
            num_classes (int, optional): Number of output classes for
                classification. Defaults to 2.
        """
        super().__init__()

        self.C_in = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        C_out: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Creates one residual stage of the network.

        Args:
            block (type[nn.Module]): Residual block class.
            C_out (int): Number of output channels for the stage.
            blocks (int): Number of residual blocks in the stage.
            stride (int, optional): Stride of the first block.
                Defaults to 1.

        Returns:
            nn.Sequential: Sequential container with residual blocks.
        """
        downsample = None

        if stride != 1 or self.C_in != C_out * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.C_in,
                    C_out * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(C_out * block.expansion),
            )

        layers = [block(self.C_in, C_out, stride, downsample)]
        self.C_in = C_out * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.C_in, C_out))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initializes convolution and batch normalization weights.

        Convolutional layers use Kaiming normal initialization.
        Batch normalization layers are initialized with unit weights
        and zero biases.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the ResNet model.

        Args:
            x (torch.Tensor): Input image tensor of shape
                ``(batch_size, 3, H, W)``.

        Returns:
            torch.Tensor: Logits tensor of shape
                ``(batch_size, num_classes)``.
        """
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
