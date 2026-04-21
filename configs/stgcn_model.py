import torch, torch.nn as nn, numpy as np

def build_A():
    num_node = 17
    edges = [(0,1),(0,2),(1,3),(2,4),(5,7),(7,9),(6,8),(8,10),
             (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    center = 11
    A = np.zeros((3, num_node, num_node), dtype=np.float32)
    for i in range(num_node): A[0,i,i] = 1
    for i,j in edges:
        di,dj = abs(i-center), abs(j-center)
        if di<dj:   A[1,j,i]=1; A[2,i,j]=1
        elif dj<di: A[1,i,j]=1; A[2,j,i]=1
        else:       A[1,i,j]=1; A[1,j,i]=1
    for p in range(3):
        s=A[p].sum(1,keepdims=True); s[s==0]=1; A[p]/=s
    return torch.tensor(A, dtype=torch.float32)

class GCNUnit(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.PA=nn.Parameter(torch.zeros_like(A))
        self.register_buffer('A', A)
        self.conv=nn.Conv2d(in_ch, out_ch*3, kernel_size=1)
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU()
    def forward(self, x):
        B,C,T,V=x.shape
        y=self.conv(x).reshape(B,3,-1,T,V)
        y=torch.einsum('bkctv,kvw->bctw', y, self.A+self.PA)
        return self.relu(self.bn(y))

class TCNUnit(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=(9,1),stride=(stride,1),padding=(4,0))
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU()
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class ResidualUnit(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=(stride,1))
        self.bn=nn.BatchNorm2d(out_ch)
    def forward(self, x): return self.bn(self.conv(x))

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, residual='identity'):
        super().__init__()
        self.gcn=GCNUnit(in_ch,out_ch,A)
        self.tcn=TCNUnit(out_ch,out_ch,stride=stride)
        self.residual_type=residual
        self.residual = ResidualUnit(in_ch,out_ch,stride) if residual=='conv' else None
        self.relu=nn.ReLU()
    def forward(self, x):
        out=self.tcn(self.gcn(x))
        if   self.residual_type=='none':     return self.relu(out)
        elif self.residual_type=='identity': return self.relu(out+x)
        else:                               return self.relu(out+self.residual(x))

class STGCN(nn.Module):
    def __init__(self, num_classes=60, in_ch=3):
        super().__init__()
        A=build_A()
        self.data_bn=nn.BatchNorm1d(3*17)
        self.gcn=nn.ModuleList([
            STGCNBlock(in_ch,64,A,stride=1,residual='none'),
            STGCNBlock(64,64,A),STGCNBlock(64,64,A),STGCNBlock(64,64,A),
            STGCNBlock(64,128,A,stride=2,residual='conv'),
            STGCNBlock(128,128,A),STGCNBlock(128,128,A),
            STGCNBlock(128,256,A,stride=2,residual='conv'),
            STGCNBlock(256,256,A),STGCNBlock(256,256,A),
        ])
        self.cls_head=nn.ModuleDict({'fc':nn.Linear(256,num_classes)})
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.drop=nn.Dropout(0.5)
    def forward(self, x):
        B,T,V,C=x.shape
        xp=x.permute(0,3,1,2).contiguous()
        xt=xp.permute(0,2,1,3).reshape(B,T,C*V).permute(0,2,1)
        self.data_bn(xt)
        out=xp
        for block in self.gcn: out=block(out)
        out=self.pool(out).squeeze(-1).squeeze(-1)
        return self.cls_head['fc'](self.drop(out))
