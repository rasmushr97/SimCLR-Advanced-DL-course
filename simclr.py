from loss import NT_Xent
import torch.optim as optim
from tqdm import tqdm
import wandb

class SimCLR():
    def __init__(self, model, device='cpu', use_wandb=True):
        self.model = model
        self.device = device
        self.model.to(device)
        self.use_wandb = use_wandb

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        self.criterion = NT_Xent(device=self.device)

    def train(self, dataloader, epochs=1):
        if self.use_wandb:
            wandb.init(project="simclr", entity="rasmushr97")
            wandb.watch(self.model, log_freq=100)

        self.model.train()

        for epoch in range(epochs):

            with tqdm(total=len(dataloader)) as t:
                t.set_description(f'Epoch: {epoch+1}/{epochs}')
                for batch_idx, (images_v1, images_v2, _) in enumerate(dataloader):
                    images_v1, images_v2 = images_v1.to(self.device), images_v2.to(self.device)

                    self.optimizer.zero_grad()

                    z1 = self.model(images_v1)
                    z2 = self.model(images_v2)

                    loss = self.criterion(z1, z2)
                    loss.backward()

                    self.optimizer.step()
                    
                    t.set_postfix(loss=loss)
                    t.update()
                    
                    if batch_idx % 100 == 0 and self.use_wandb:
                        wandb.log({"loss": loss})


