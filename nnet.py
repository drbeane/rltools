import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class NNet(nn.Module):

    def __init__(self, model):
        super(NNet, self).__init__()
        self.model = model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        self.history = {
            'train_loss' : [],
            'train_acc' : [],
            'valid_loss' : [],
            'valid_acc' : []
        }

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        output = self.model(x)
        return output

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x).numpy()
            probs = nn.Softmax(logits, dim=-1).numpy()
        return probs

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x).numpy()
            probs = nn.Softmax(logits, dim=-1)
        return np.argmax(probs, axis=1)

    def train_model(self, X, y, epochs, batch_size, lr, val_split, seed, updates=1):
        import math
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        X_tensor = torch.autograd.Variable(torch.FloatTensor(X))
        y_tensor = torch.autograd.Variable(torch.LongTensor(y))

        num_va_obs = int(val_split * len(y))
        num_tr_obs = len(y) - num_va_obs

        X_train = X_tensor[:num_tr_obs, :]
        X_valid = X_tensor[num_tr_obs:, :]
        y_train = y_tensor[:num_tr_obs]
        y_valid = y_tensor[num_tr_obs:]

        num_tr_batches = math.ceil(len(y_train) / batch_size)
        num_va_batches = math.ceil(len(y_valid) / batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for n in range(epochs):

            #----------------------------
            # Training Loop
            #----------------------------
            self.model.train()
            train_loss = 0
            train_correct = 0
            for i in range(num_tr_batches):
                # Get new batch
                start = i * batch_size
                end   = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # Calculate output, then loss
                optimizer.zero_grad()
                output = self.model.forward(X_batch)
                loss = loss_fn(output, y_batch)

                # Perform Update
                loss.backward()
                optimizer.step()

                # Add batch loss to epoch loss, weighting by batch size
                train_loss += loss.item() * len(y_batch)

                # Count correct predictions
                values, labels = torch.max(output, 1)
                train_correct += np.sum(labels.cpu().numpy() == y_batch.cpu().numpy())

            #----------------------------
            # Validation Loop
            #----------------------------
            self.model.eval()
            valid_loss = 0
            valid_correct = 0
            for i in range(num_va_batches):
                # Get new batch
                start = i * batch_size
                end   = start + batch_size
                X_batch = X_valid[start:end]
                y_batch = y_valid[start:end]

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # Calculate output, then loss
                with torch.no_grad():
                    output = self.model(X_batch)
                    loss = loss_fn(output, y_batch)

                # Add batch loss to epoch loss, weighting by batch size
                valid_loss += loss.item() * len(y_batch)

                # Count correct predictions
                values, labels = torch.max(output, 1)
                valid_correct += np.sum(labels.cpu().numpy() == y_batch.cpu().numpy())

            train_loss /= len(y_train)
            train_acc = train_correct / len(y_train)

            valid_loss /= len(y_valid)
            valid_acc = valid_correct / len(y_valid)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_acc'].append(valid_acc)

            if (n+1) % updates == 0:
                print(f'Epoch {n+1}: Training loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}, Val Loss: {valid_loss:.4f}, Val Acc {valid_acc:.4f}')

    def training_curves(self, start=1, figsize=[12,4]):
        epoch_range = range(start, len(self.history['train_loss'])+1)
        plt.figure(figsize=figsize)
        plt.subplot(1,2,1)
        plt.plot(epoch_range, self.history['train_loss'][start-1:], label='Training', zorder=2)
        if len(self.history['train_loss']) > 0:
            plt.plot(epoch_range, self.history['valid_loss'][start-1:], label='Validation', zorder=2)
        plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Loss by Epoch')
        plt.grid()
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(epoch_range, self.history['train_acc'][start-1:], label='Training', zorder=2)
        if len(self.history['train_acc']) > 0:
            plt.plot(epoch_range, self.history['valid_acc'][start-1:], label='Validation', zorder=2)
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.title('Accuracy by Epoch')
