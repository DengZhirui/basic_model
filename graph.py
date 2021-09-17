from torch_geometric.nn import GCNConv, GATConv, SAGEConv
class Graph_Network(nn.Module):
    def __init__(self, in_feature, hid1_feature, gcnout_feature, device, dropout=0):
        super(Graph_Network, self).__init__()
        self.device = device
        self.dropout = dropout
        self.conv1 = SAGEConv(in_feature, hid1_feature)
        self.conv2 = SAGEConv(hid1_feature, gcnout_feature)
        #self.diag = nn.Parameter(torch.FloatTensor(hid2_feature), requires_grad=True)
        #init.xavier_normal_(self.diag)
        self.linear = nn.Linear(gcnout_feature, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #A = torch.mm(x, torch.t(x))#.to_sparse() # 可以试试用cos距离
        #A = torch.mul(torch.mm(torch.mm(x, torch.diag(self.diag)),torch.t(x)), 1-torch.eye(x.shape[0]))
        #A = torch.mul((F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)+1)/2, (1-torch.eye(x.shape[0])))
        x = F.dropout(x, self.dropout)
        hid = x
        x = self.conv2(x, edge_index) # [node_num, 1]

        #x = F.relu(x)
        #temp = torch.ge(x.squeeze(1), 0.5).float().repeat(x.shape[0], 1)
        #A = torch.mm(torch.t(temp), torch.mm(temp, A))
        #x = self.mlp(x)
        #x = torch.sigmoid(x)
        A = 0
        return x, A, hid
        #return F.log_softmax(x, dim=1)