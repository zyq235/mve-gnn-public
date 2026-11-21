from models.mtgnn.layer import *

class GTNetCore(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1,
                 conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128,
                 seq_length=12, in_dim=1, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(GTNetCore, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(
                    nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # Dynamic Skip Connection Calculation
                if self.seq_length > self.receptive_field:
                    skip_kernel_w = self.seq_length - rf_size_j + 1
                else:
                    skip_kernel_w = self.receptive_field - rf_size_j + 1

                self.skip_convs.append(
                    nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, skip_kernel_w)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.idx = torch.arange(self.num_nodes).to(device)

        # Adapt skip connections for output
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        # Remove last conv to return features instead of predictions
        # self.end_conv_2 = ... 

    def forward(self, input):
        # Input shape: (B, C, N, T)
        seq_len = input.size(3)
        assert seq_len == self.seq_length, f'Input sequence length {seq_len} not equal to preset sequence length {self.seq_length}'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            adp = self.gc(self.idx)
        else:
            adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))  # (B, End_Channels, N, 1)
        return x


# ==========================================
# Part 3: Classifier Wrapper
# ==========================================

class GTNetClassifier(nn.Module):
    def __init__(self,
                 num_nodes=131,
                 seq_len=24,
                 in_dim=1,
                 out_dim=256,
                 num_classes=2,
                 dropout=0.3,
                 device='cuda',
                 # GTNet specific hyperparams
                 layers=3,
                 gcn_depth=2,
                 conv_channels=32,
                 residual_channels=32,
                 skip_channels=64,
                 end_channels=128):
        super(GTNetClassifier, self).__init__()

        self.device = device

        # Initialize GTNet Core
        self.gtnet = GTNetCore(
            gcn_true=True,
            buildA_true=True,
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            device=device,
            dropout=dropout,
            subgraph_size=20,
            node_dim=40,
            dilation_exponential=2,
            conv_channels=conv_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            seq_length=seq_len,
            in_dim=in_dim,
            layers=layers,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True
        )

        # Classification Head
        # GTNet output shape: (B, end_channels, N, 1)
        # We pool over N (Nodes)
        self.classifier = nn.Sequential(
            nn.Linear(end_channels, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x, demo):
        """
        x: (Batch, 129, Seq_Len) - Time Series
        demo: (Batch, 2) - Static Demographics
        """
        # 1. Combine TS and Demo into Graph Nodes
        # Expand demo to match time dimension: (B, 2) -> (B, 2, Seq_Len)
        demo_expanded = demo.unsqueeze(-1).expand(-1, -1, x.size(2))

        # Concatenate: (B, 129+2, T) = (B, 131, T)
        x_combined = torch.cat([x, demo_expanded], dim=1)

        # 2. Reshape for GTNet: (B, In_Dim, Nodes, T) -> (B, 1, 131, T)
        x_input = x_combined.unsqueeze(1)

        # 3. Pass through GTNet
        # Output: (B, End_Channels, 131, 1)
        features = self.gtnet(x_input)

        # 4. Pooling
        # Squeeze time dim (which is 1) -> (B, End_Channels, 131)
        features = features.squeeze(-1)
        # Global Average Pooling over Nodes -> (B, End_Channels)
        graph_embedding = F.avg_pool1d(features, features.size(2)).squeeze(2)

        # 5. Classification
        logits = self.classifier(graph_embedding)

        return logits