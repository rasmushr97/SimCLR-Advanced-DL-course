from loss import NT_Xent
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from collections import deque

class SimCLR():
    def __init__(self, model, tau=0.5, device='cpu', use_wandb=False, log_iterval=10, save_after_epoch=False):
        self.model = model
        self.device = device
        self.model.to(device)
        self.use_wandb = use_wandb
        self.log_iterval = log_iterval
        self.save_after_epoch = save_after_epoch

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        self.criterion = NT_Xent(temperature=tau, device=self.device)

    def train(self, dataloader, epochs=1):
        if self.use_wandb:
            wandb.init(project="simclr", entity="rasmushr97")
            wandb.watch(self.model, log_freq=100)

        self.model.train()

        for epoch in range(epochs):
            running_loss = deque(maxlen=20)
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
                    
                    running_loss.append(loss.item())
                    loss_avg = sum(running_loss) / len(running_loss)
                    t.set_postfix(loss=loss_avg)
                    t.update()
                    
                    if batch_idx % self.log_iterval == 0 and self.use_wandb:
                        wandb.log({"loss": loss})

            if self.save_after_epoch:
                torch.save(model, f'model-{epoch}.h5')

                if self.use_wandb:
                    wandb.save(f'model-{epoch}.h5')



