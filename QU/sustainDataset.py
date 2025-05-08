

class SustainDataset(Dataset):
    '''Klasse um Dataset für custom Bert Modell vorzubereiten.'''
    def __init__(self, encodings, additional_features=None, labels=None, super_labels=None):
        self.encodings = encodings
        self.additional_features = additional_features
        self.labels = labels
        self.super_labels = super_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        if self.additional_features is not None:
            item['additional_features'] = torch.tensor(self.additional_features[idx], dtype=torch.float)
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.super_labels is not None:
            item['super_labels'] = torch.tensor(self.super_labels[idx], dtype=torch.long)
        
        return item

    def __len__(self):
        if self.labels is not None:
            return len(self.labels)
        elif self.additional_features is not None:
            return len(self.additional_features)
        else:
            return len(next(iter(self.encodings.values())))
