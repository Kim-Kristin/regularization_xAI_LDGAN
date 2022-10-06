import torch

# Nutzen der GPU wenn vorhanden, ansonsten CPU


def get_default_device():
    if torch.cuda.is_available():     # Wenn cuda verfügbar dann:
        return torch.device('cuda')   # Nutze Device = Cuda (=GPU)
    else:                         # Ansonsten
        return torch.device('cpu')    # Nutze Device = CPU


# Anzeigen welches Device verfügbar ist
device = get_default_device()
print(device)

# Hilfsklasse zum Verschieben des Dataloaders "org_loader" auf das jeweilige Device


class DeviceDataLoader():

    # Initialisierung
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    # Anzahl der Images pro Batch
    def __len__(self):
        return len(self.dataloader)

    # Erstellt einen Batch an Tensoren nach dem Verschieben auf das Device
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(tensor.to(self.device) for tensor in batch)
