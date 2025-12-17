import torch
import torch.nn as nn


from binary_layers.Layers import ResidualBCVNBlock, BNCVL, BinaryLinear
from real_layers.RealLayers import RealResidualCVNBlock, RealCVL, RealLinear


class BCVNN(nn.Module):
    def __init__(self, image_channels=3, filter_dimension=3, num_classes=101):
        """
        image_channels: number of input channels (3 for RGB)
        filter_dimension: kernel size for all convolution layers
        """
        super().__init__()

        # block 1
        self.block1 = ResidualBCVNBlock(
            [
                {"in_channels": image_channels, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 32, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 2
        self.block2 = ResidualBCVNBlock(
            [
                {"in_channels": 32, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 3
        self.block3 = ResidualBCVNBlock(
            [
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 4
        self.block4 = ResidualBCVNBlock(
            [
                {"in_channels": 64, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 128, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 5
        self.bncvl9 = BNCVL(in_channels=128, out_channels=256, kernel_size=filter_dimension, activation="relu", padding="same")
        self.bncvl10 = BNCVL(in_channels=256, out_channels=256, kernel_size=filter_dimension, activation="relu", padding="same")
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        # block 6
        self.bnfcl1 = BinaryLinear(in_features=256, out_features=256, activation="relu")
        self.bnfcl2 = BinaryLinear(in_features=256, out_features=256, activation="relu")
        self.finallayer = BinaryLinear(in_features=256, out_features=num_classes, activation=None)
    
    def forward(self, x):
        # --- feature extraction ---
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # --- final conv + normalization + activation ---
        x = self.bncvl9(x)
        x = self.bncvl10(x)

        # --- global average pooling ---
        x = self.GAP(x)          # shape: [batch, channels, 1, 1]
        x = torch.flatten(x, 1)  # shape: [batch, channels]

        # --- fully connected binary layers ---
        x = self.bnfcl1(x)
        x = self.bnfcl2(x)
        x = self.finallayer(x)

        return x



class RealCVNN(nn.Module):
    """
    Non-binary equivalent of BCVNN
    """
    def __init__(self, image_channels=3, filter_dimension=3, num_classes=101):
        super().__init__()

        # block 1
        self.block1 = RealResidualCVNBlock(
            [
                {"in_channels": image_channels, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 32, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 2
        self.block2 = RealResidualCVNBlock(
            [
                {"in_channels": 32, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 3
        self.block3 = RealResidualCVNBlock(
            [
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 4
        self.block4 = RealResidualCVNBlock(
            [
                {"in_channels": 64, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 128, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 5
        self.conv9 = RealCVL(in_channels=128, out_channels=256, kernel_size=filter_dimension, activation="relu", padding="same")
        self.conv10 = RealCVL(in_channels=256, out_channels=256, kernel_size=filter_dimension, activation="relu", padding="same")
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        # block 6
        self.fc1 = RealLinear(in_features=256, out_features=256, activation="relu", use_layernorm=False)
        self.fc2 = RealLinear(in_features=256, out_features=256, activation="relu", use_layernorm=False)
        self.final = RealLinear(in_features=256, out_features=num_classes, activation=None, use_layernorm=False)

    def forward(self, x):
        # --- feature extraction ---
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # --- final conv + normalization + activation ---
        x = self.conv9(x)
        x = self.conv10(x)

        # --- global average pooling ---
        x = self.GAP(x)          # shape: [batch, 256, 1, 1]
        x = torch.flatten(x, 1)  # shape: [batch, 256]

        # --- fully connected binary layers ---
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.final(x)
        return x
    

class ResidualConvGAPBlock(nn.Module):
    """
    Residual block with two conv layers (conv9, conv10) that can independently
    be binary or real. Uses global average pooling (GAP) at the end.
    """
    def __init__(self, kind1, kind2, in_c, mid_c, filter_dim=3):
        super().__init__()

        # conv9
        if kind1 == "binary":
            self.conv1 = BNCVL(in_channels=in_c, out_channels=mid_c, kernel_size=filter_dim,
                               activation="relu", padding="same")
        elif kind1 == "real":
            self.conv1 = RealCVL(in_channels=in_c, out_channels=mid_c, kernel_size=filter_dim,
                                 activation="relu", padding="same")
        else:
            raise ValueError(f"Unknown conv type for conv9: {kind1}")

        # conv10
        if kind2 == "binary":
            self.conv2 = BNCVL(in_channels=mid_c, out_channels=mid_c, kernel_size=filter_dim,
                               activation="relu", padding="same")
        elif kind2 == "real":
            self.conv2 = RealCVL(in_channels=mid_c, out_channels=mid_c, kernel_size=filter_dim,
                                 activation="relu", padding="same")
        else:
            raise ValueError(f"Unknown conv type for conv10: {kind2}")

        # 1x1 projection if needed
        if in_c != mid_c:
            self.proj = nn.Conv2d(in_c, mid_c, kernel_size=1, stride=1, padding=0, bias=False)
            self.proj_bn = nn.BatchNorm2d(mid_c)
        else:
            self.proj = nn.Identity()
            self.proj_bn = None

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity = x
        if not isinstance(self.proj, nn.Identity):
            identity = self.proj(identity)
            identity = self.proj_bn(identity)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.pool(out)
        return out
    


class ResidualConvMAXBlock(nn.Module):
    """
    Residual block with two conv layers that can independently
    be binary or real. Uses MaxPool2d at the end.
    """
    def __init__(self, kind1, kind2, in_c, mid_c, filter_dim=3):
        super().__init__()

        # conv
        if kind1 == "binary":
            self.conv1 = BNCVL(in_channels=in_c, out_channels=mid_c, kernel_size=filter_dim,
                               activation="relu", padding="same")
        elif kind1 == "real":
            self.conv1 = RealCVL(in_channels=in_c, out_channels=mid_c, kernel_size=filter_dim,
                                 activation="relu", padding="same")
        else:
            raise ValueError(f"Unknown conv type: {kind1}")

        # conv
        if kind2 == "binary":
            self.conv2 = BNCVL(in_channels=mid_c, out_channels=mid_c, kernel_size=filter_dim,
                               activation="relu", padding="same")
        elif kind2 == "real":
            self.conv2 = RealCVL(in_channels=mid_c, out_channels=mid_c, kernel_size=filter_dim,
                                 activation="relu", padding="same")
        else:
            raise ValueError(f"Unknown conv type: {kind2}")

        # 1x1 projection if needed
        if in_c != mid_c:
            self.proj = nn.Conv2d(in_c, mid_c, kernel_size=1, stride=1, padding=0, bias=False)
            self.proj_bn = nn.BatchNorm2d(mid_c)
        else:
            self.proj = nn.Identity()
            self.proj_bn = None

        # Max pooling 2d
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x
        if not isinstance(self.proj, nn.Identity):
            identity = self.proj(identity)
            identity = self.proj_bn(identity)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.pool(out)
        return out



# class MixedCVNN(nn.Module):
#     def __init__(self, model_config, image_channels=3, filter_dimension=3, num_classes=101):
#         super().__init__()

#         # utility helpers
#         def choose_residual(kind, layers):
#             if kind == "binary":
#                 return ResidualBCVNBlock(layers)
#             elif kind == "real":
#                 return RealResidualCVNBlock(layers)
#             else:
#                 raise ValueError(f"Unknown block type: {kind}")

#         def choose_conv(kind, in_c, out_c):
#             if kind == "binary":
#                 return BNCVL(in_channels=in_c, out_channels=out_c,
#                              kernel_size=filter_dimension, activation="relu", padding="same")
#             elif kind == "real":
#                 return RealCVL(in_channels=in_c, out_channels=out_c,
#                                kernel_size=filter_dimension, activation="relu", padding="same")
#             else:
#                 raise ValueError(f"Unknown conv type: {kind}")

#         def choose_linear(kind, in_f, out_f, activation="relu", use_layernorm=False):
#             if kind == "binary":
#                 return BinaryLinear(in_features=in_f, out_features=out_f, activation=activation)
#             elif kind == "real":
#                 return RealLinear(in_features=in_f, out_features=out_f, 
#                                   activation=activation, use_layernorm=use_layernorm)
#             else:
#                 raise ValueError(f"Unknown linear type: {kind}")

#         # ------------------ blocks 1–4 ------------------

#         self.block1 = choose_residual(
#             model_config["block1"],
#             [
#                 {"in_channels": image_channels, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#                 {"in_channels": 32, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#             ]
#         )

#         self.block2 = choose_residual(
#             model_config["block2"],
#             [
#                 {"in_channels": 32, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#                 {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#             ]
#         )

#         self.block3 = choose_residual(
#             model_config["block3"],
#             [
#                 {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#                 {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#             ]
#         )

#         self.block4 = choose_residual(
#             model_config["block4"],
#             [
#                 {"in_channels": 64, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#                 {"in_channels": 128, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
#             ]
#         )

#         # ------------------ conv layers ------------------

#         # self.conv9 = choose_conv(model_config["conv9"], 128, 256)
#         # self.conv10 = choose_conv(model_config["conv10"], 256, 256)
#         # Replacing the old conv layers with a Block that includes GAP and residual connection
#         self.conv_block = ResidualConvGAPBlock(
#             kind1=model_config["conv9"],
#             kind2=model_config["conv10"],
#             in_c=128,
#             mid_c=256,
#             filter_dim=filter_dimension
#         )

#         self.GAP = nn.AdaptiveAvgPool2d((1, 1)) # not used as the previous block already has it

#         # ------------------ fully connected layers ------------------

#         # real FC layers allow disabling layernorm, binary does not
#         self.fc1 = choose_linear(model_config["fc1"], 256, 256, activation="relu")
#         self.fc2 = choose_linear(model_config["fc2"], 256, 256, activation="relu")

#         # final layer has activation=None
#         self.final = choose_linear(model_config["final"], 256, num_classes, activation=None)


#     def forward(self, x):
#         # --- feature extraction ---
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)

#         # --- final conv + normalization + activation ---
#         # x = self.conv9(x)
#         # x = self.conv10(x)
#         x = self.conv_block(x)

#         # --- global average pooling ---
#         # x = self.GAP(x)          # shape: [batch, 256, 1, 1]
#         x = torch.flatten(x, 1)  # shape: [batch, 256]

#         # --- fully connected layers ---
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.final(x)
#         return x
    


class MixedCVNN(nn.Module):
    def __init__(self, model_config, image_channels=3, filter_dimension=3, num_classes=101):
        super().__init__()

        # ------------------ helpers ------------------

        def choose_conv(kind, in_c, out_c):
            if kind == "binary":
                return BNCVL(in_channels=in_c, out_channels=out_c,
                             kernel_size=filter_dimension, activation="relu", padding="same")
            elif kind == "real":
                return RealCVL(in_channels=in_c, out_channels=out_c,
                               kernel_size=filter_dimension, activation="relu", padding="same")
            else:
                raise ValueError(f"Unknown conv type: {kind}")

        def choose_linear(kind, in_f, out_f, activation="relu", use_layernorm=False):
            if kind == "binary":
                return BinaryLinear(in_features=in_f, out_features=out_f, activation=activation)
            elif kind == "real":
                return RealLinear(in_features=in_f, out_features=out_f,
                                  activation=activation, use_layernorm=use_layernorm)
            else:
                raise ValueError(f"Unknown linear type: {kind}")

        # this uses the more flexible ResidualConvMAXBlock

        # block1: 3 → 32
        self.block1 = ResidualConvMAXBlock(
            kind1=model_config["conv1"],   # "binary" or "real"
            kind2=model_config["conv2"],
            in_c=image_channels,
            mid_c=32,
            filter_dim=filter_dimension
        )

        # block2: 32 → 64
        self.block2 = ResidualConvMAXBlock(
            kind1=model_config["conv3"],
            kind2=model_config["conv4"],
            in_c=32,
            mid_c=64,
            filter_dim=filter_dimension
        )

        # block3: 64 → 64
        self.block3 = ResidualConvMAXBlock(
            kind1=model_config["conv5"],
            kind2=model_config["conv6"],
            in_c=64,
            mid_c=64,
            filter_dim=filter_dimension
        )

        # block4: 64 → 128
        self.block4 = ResidualConvMAXBlock(
            kind1=model_config["conv7"],
            kind2=model_config["conv8"],
            in_c=64,
            mid_c=128,
            filter_dim=filter_dimension
        )

        # ------------------ conv9/conv10 combined block ------------------

        self.conv_block = ResidualConvGAPBlock(
            kind1=model_config["conv9"],
            kind2=model_config["conv10"],
            in_c=128,
            mid_c=256,
            filter_dim=filter_dimension
        )

        # ------------------ fully connected ------------------

        self.fc1 = choose_linear(model_config["fc1"], 256, 256, activation="relu")
        self.fc2 = choose_linear(model_config["fc2"], 256, 256, activation="relu")
        self.final = choose_linear(model_config["final"], 256, num_classes, activation=None)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.conv_block(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.final(x)
        return x
