import torch.nn as nn
import torch.nn.functional as F
import torch

class NT_Xent(nn.Module):
    def __init__(self, temperature=1.0, device='cpu'):
        super(NT_Xent, self).__init__()
        self.tau = temperature
        self.device = device

    def forward(self, z1, z2):
        assert len(z1) == len(z2)

        N = 2 * len(z1)

        # Cosine similarity
        z1, z2 = F.normalize(z1), F.normalize(z2) # Normalized to compute cosine similiarty
        z = torch.cat((z1, z2), dim=0)
        sim = torch.exp(torch.mm(z, z.t()) / self.tau) #

        # Compute negative similarity by masking out diagonal and summing rows
        mask = ~torch.eye(N, device=self.device).bool()
        neg = sim.masked_select(mask).view(N, -1)
        neg = torch.sum(neg, dim=-1)

        # Compute positive similarity by dot product of normalized outputs
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / self.tau)
        pos = torch.cat([pos, pos])

        loss = -torch.log(pos / neg)
        loss = torch.mean(loss)
        
        return loss
