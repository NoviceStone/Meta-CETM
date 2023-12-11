import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    """returns a block conv-bn-relu-pool
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, (3,), padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2, 2),
    )


def mlp_block(in_features, out_features):
    """returns a block linear-bn-relu
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU()
    )


class ClassificationNet(nn.Module):

    def __init__(self, input_dim, num_classes, arch='mlp'):
        super(ClassificationNet, self).__init__()
        if arch == 'conv':
            self.encoder = nn.Sequential(
                conv_block(1, 2),
                conv_block(2, 4),
                conv_block(4, 4),
                nn.Flatten(),
            )
            # flatten dim: 2984 for 20ng, 3752 for yahoo, 3136 for dbpedia, 2460 for wos
            self.classifier = nn.Linear(2984, num_classes)
        else:
            self.encoder = nn.Sequential(
                mlp_block(input_dim, 512),
                mlp_block(512, 256),
                mlp_block(256, 128),
            )
            self.classifier = nn.Linear(128, num_classes)
        self.use_conv = True if arch == 'conv' else False

    def forward(self, x, fast_weights=None):
        if self.use_conv:
            x = x.unsqueeze(1)
        if fast_weights is None:
            h = self.encoder(x)
            out = self.classifier(h)
        else:
            # implement MAML forward for mlp architecture
            fast_weights = {name: fast_weights[i] for i, (name, _) in enumerate(self.named_parameters())}
            h = F.relu(F.batch_norm(
                F.linear(x, fast_weights['encoder.0.0.weight'], fast_weights['encoder.0.0.bias']),
                running_mean=None, running_var=None,
                weight=fast_weights['encoder.0.1.weight'],
                bias=fast_weights['encoder.0.1.bias'],
                training=True
            ))
            h = F.relu(F.batch_norm(
                F.linear(h, fast_weights['encoder.1.0.weight'], fast_weights['encoder.1.0.bias']),
                running_mean=None, running_var=None,
                weight=fast_weights['encoder.1.1.weight'],
                bias=fast_weights['encoder.1.1.bias'],
                training=True
            ))
            h = F.relu(F.batch_norm(
                F.linear(h, fast_weights['encoder.2.0.weight'], fast_weights['encoder.2.0.bias']),
                running_mean=None, running_var=None,
                weight=fast_weights['encoder.2.1.weight'],
                bias=fast_weights['encoder.2.1.bias'],
                training=True
            ))
            out = F.linear(h, fast_weights['classifier.weight'], fast_weights['classifier.bias'])
        return h, out
