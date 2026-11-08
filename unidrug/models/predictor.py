import torch.nn as nn

activation = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}

class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        hidden_dims = [input_dim] + config["hidden_size"] + [output_dim]
        for i in range(len(hidden_dims) - 1):
            self.model.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != len(hidden_dims) - 2:
                self.model.append(nn.Dropout(config["dropout"]))
                if config["activation"] != "none":
                    self.model.append(activation[config["activation"]])
                if config["batch_norm"]:
                    self.model.append(nn.BatchNorm1d(hidden_dims[i + 1]))
    
    def forward(self, h):
        output = self.model(h)
        # 只有在批次大小>1时才squeeze，避免批次大小为1时的形状不匹配
        if output.shape[0] > 1:
            return output.squeeze()
        else:
            return output