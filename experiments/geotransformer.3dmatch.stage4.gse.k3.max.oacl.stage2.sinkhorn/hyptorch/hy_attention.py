# from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F
# import geotorch

class MetricTensor(nn.Module):
    def __init__(self, hidden_features):
        super().__init__()
        self.M = torch.zeros([hidden_features,hidden_features],dtype=torch.float32,requires_grad=True)
        self.M = nn.Parameter(self.M, requires_grad=True)

    def forward(self, h, h_transpose):
        out = h  @ self.M @ h_transpose

        return out



class HypAttention(nn.Module):

    # graph attention layer with learnable matrix

    def __init__(self, 
        in_features, 
        hidden_features,
        n_heads=8,
        num_neighbors=None):
        super(HypAttention, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.n_heads = n_heads


        self.W_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.M = MetricTensor(hidden_features)
        

        self.W = nn.Linear(in_features,in_features,True)


        self.norm = nn.LayerNorm(in_features)
        self.activation = nn.ReLU(True)

    def forward(self, h):


        b, num_centroids, c = h.shape

        h_org = h


        h = self.W(h_org)

        h = h.view(b, num_centroids, self.n_heads, -1)
        h = h.permute(0,2,1,3) # [b, n_heads, num_centroids, c]
        attention = self.M(h, h.transpose(-2,-1)) 

        attention = F.softmax(attention, -1)
        h = attention @ h

        out = h
        out = out.permute(0,2,1,3)
        out = out.contiguous()
        out = out.view(b, num_centroids, -1)
        

        out = self.norm(out)
        out = self.activation(out)


        return out


class HypCrossAttention(nn.Module):

    # graph attention layer with learnable matrix

    def __init__(self, 
        in_features, 
        hidden_features,
        n_heads=8,
        num_neighbors=None):
        super(HypCrossAttention, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.n_heads = n_heads

        self.M = MetricTensor(hidden_features)
        

        self.W = nn.Linear(in_features,in_features,True)
        self.W_key = nn.Linear(in_features,in_features,True)
        self.W_value = nn.Linear(in_features,in_features,True)


        self.norm = nn.LayerNorm(in_features)
        self.activation = nn.ReLU(True)

    def forward(self, query, key, value):

        b, num_centroids, c = query.shape
        b1, num_centroids1, c1 = key.shape


        h = self.W(query)
        h = h.view(b, num_centroids, self.n_heads, -1)
        h = h.permute(0,2,1,3) # [b, n_heads, num_centroids, c]

        key = self.W_key(key)
        key = key.view(b1, num_centroids1, self.n_heads, -1)
        key = key.permute(0,2,1,3)

        value = self.W_value(value)
        value = value.view(b1, num_centroids1, self.n_heads, -1)
        value = value.permute(0,2,1,3)

        attention = self.M(h, key.transpose(-2,-1)) 

        attention = F.softmax(attention, -1)
        h = attention @ value

        out = h
        out = out.permute(0,2,1,3)
        out = out.contiguous()
        out = out.view(b, num_centroids, -1)
        

        out = self.norm(out)
        out = self.activation(out)

        return out