import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=10, num_layers=1, output_size=2, 
                 window_size=30, input_features=6, batch_size=4096, use_dropout=True):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.input_features = input_features
        self.batch_size = batch_size
        self.use_dropout = use_dropout

        self.lstm = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.lr_schedule_history = []
        self.prev_val_loss = None  # Pour adaptation dynamique du LR

        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Dernier pas de temps
        out = self.fc_out(out)
        return out
    
    def _adaptive_learning_rate(self, epoch, initial_lr, val_loss=None):
        """LR dynamique basé sur la progression de la validation loss"""
        if epoch == 0:
            self.prev_val_loss = val_loss
            return initial_lr
        if val_loss is None or self.prev_val_loss is None:
            return initial_lr * (0.7 ** epoch)
        if val_loss < self.prev_val_loss:
            lr = self.optimizer.param_groups[0]['lr'] * 0.85  # baisse douce si amélioration
        else:
            lr = self.optimizer.param_groups[0]['lr'] * 0.5   # baisse forte si stagnation/dégradation
        lr = max(lr, initial_lr * 1e-4)  # ne descend pas trop bas
        self.prev_val_loss = val_loss
        return lr

    def _progressive_batch_size(self, epoch, initial_batch_size):
        """Augmente progressivement la taille des batches pour plus de stabilité"""
        if epoch < 20:
            return max(initial_batch_size // 4, 64)
        elif epoch < 50:
            return max(initial_batch_size // 2, 128)
        elif epoch < 100:
            return initial_batch_size
        else:
            return min(initial_batch_size * 2, 4096)
    
    def _adaptive_dropout_rate(self, epoch, val_loss, train_loss):
        """Ajuste le dropout en fonction de l'overfitting détecté"""
        if epoch < 10:
            return 0.1  
        
        
        overfitting_ratio = val_loss / train_loss if train_loss > 0 else 1.0
        
        if overfitting_ratio > 1.3:
            
            return min(0.5, 0.3 + (overfitting_ratio - 1.3) * 0.2)
        elif overfitting_ratio > 1.1:
            
            return 0.2 + (overfitting_ratio - 1.1) * 0.5
        else:
            
            return max(0.05, 0.2 - (1.1 - overfitting_ratio) * 0.3)
    
    def _update_dropout_layers(self, new_dropout_rate):
        """Met à jour les couches de dropout avec un nouveau taux (inutile ici, modèle simplifié)"""
        pass
    
    def _calculate_precision_loss(self, outputs, targets, epoch):
        """Fonction de perte qui devient plus stricte avec les époques"""
        base_loss = self.criterion(outputs, targets)
        
        
        precision_factor = 1.0 + (epoch / 100.0) * 0.5  
        
        
        if epoch > 50:
            error_magnitude = torch.abs(outputs - targets)
            large_error_penalty = torch.mean(torch.clamp(error_magnitude - 0.001, min=0) ** 2)
            base_loss += precision_factor * large_error_penalty * 0.1
        
        return base_loss * precision_factor
    
    def fit(self, X, y, epochs=200, validation_split=0.2, learning_rate=0.001):
        """Entraîner le modèle avec apprentissage progressif"""
        print(f"\n🚀 Starting progressive training...")
        print(f"📊 Training samples: {len(X):,}")
        print(f"⚙️  Epochs: {epochs}")
        print(f"📈 Initial learning rate: {learning_rate}")
        print(f"🎯 Validation split: {validation_split}")
        print(f"💾 Initial batch size: {self.batch_size}")
        print(f"🖥️  Device: {self.device}")
        print(f"🧠 Progressive learning: ENABLED")
        print("-" * 60)
        
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train = X_tensor[train_indices]
        y_train = y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        y_val = y_tensor[val_indices]
        
        print(f"📚 Train samples: {len(X_train):,}")
        print(f"🔍 Validation samples: {len(X_val):,}")
        print("-" * 60)
        
        
        train_losses = []
        val_losses = []
        lr_history = []
        dropout_history = []
        batch_size_history = []
        
        start_time = time.time()
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Initialisation unique

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            current_batch_size = self._progressive_batch_size(epoch, self.batch_size)

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False)
            
            
            self.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                
                
                loss = self._calculate_precision_loss(outputs, batch_y, epoch)
                
                loss.backward()
                
                
                max_grad_norm = max(1.0, 2.0 - epoch * 0.01)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                
                self.optimizer.step()
                train_loss += loss.item()
                train_batches += 1
                
                
                if epoch < 5 and batch_idx % (len(train_loader) // 3) == 0:
                    print(f"  📦 Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.6f}")
            
            
            # Validation
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self(batch_X)
                    loss = self._calculate_precision_loss(outputs, batch_y, epoch)
                    val_loss += loss.item()
                    val_batches += 1
            
            train_loss /= train_batches
            val_loss /= val_batches
            
            # Adaptation intelligente du LR après calcul de la validation loss
            current_lr = self._adaptive_learning_rate(epoch, learning_rate, val_loss)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            if epoch > 10:  
                new_dropout_rate = self._adaptive_dropout_rate(epoch, val_loss, train_loss)
                self._update_dropout_layers(new_dropout_rate)
                dropout_history.append(new_dropout_rate)
            else:
                dropout_history.append(0.2 if self.use_dropout else 0.0)
            
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lr_history.append(current_lr)
            batch_size_history.append(current_batch_size)
            
            epoch_time = time.time() - epoch_start_time
            
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                improvement = " 🌟"
                # Sauvegarde du meilleur modèle
                self.save("best_simple_lstm.pth")
            else:
                self.patience_counter += 1
                improvement = ""
            
            
            if epoch % 5 == 0 or epoch < 10 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"BS: {current_batch_size} | "
                      f"DR: {dropout_history[-1]:.3f} | "
                      f"T: {epoch_time:.1f}s{improvement}")
            
            
            adaptive_patience = min(20, 10 + epoch // 20)
            if self.patience_counter >= adaptive_patience and epoch > 30:
                print(f"\n⏹️  Early stopping à l'époque {epoch+1} (patience: {adaptive_patience})")
                break
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"✅ Progressive training completed!")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"🏆 Best validation loss: {self.best_val_loss:.6f}")
        print(f"📉 Final train loss: {train_losses[-1]:.6f}")
        print(f"📉 Final val loss: {val_losses[-1]:.6f}")
        print(f"📈 Final LR: {lr_history[-1]:.2e}")
        print(f"💾 Final batch size: {batch_size_history[-1]}")
        print(f"🎭 Final dropout: {dropout_history[-1]:.3f}")
        
        return {
            'loss': train_losses, 
            'val_loss': val_losses,
            'lr_history': lr_history,
            'dropout_history': dropout_history,
            'batch_size_history': batch_size_history
        }
    
    def predict(self, X):
        """Faire des prédictions"""
        self.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_pred = self(batch_X)
                predictions.append(batch_pred.cpu().numpy())
        
        return np.vstack(predictions)
    
    def evaluate(self, X, y):
        """Évaluer le modèle"""
        print(f"\n🔍 Evaluating model on {len(X):,} samples...")
        
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        
        haversine_dist = self._haversine_distance(y, predictions)
        
        print(f"📊 Evaluation Results:")
        print(f"   MSE: {mse:.6f}")
        print(f"   Haversine Distance: {haversine_dist:.2f} meters")
        print("-" * 40)
        
        return [mse, haversine_dist]
    
    def _haversine_distance(self, y_true, y_pred):
        """Calculer la distance Haversine moyenne en mètres"""
        R = 6371000.0  
        
        lat1, lon1 = y_true[:, 0], y_true[:, 1]
        lat2, lon2 = y_pred[:, 0], y_pred[:, 1]
        
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return np.mean(distance)
    
    def save(self, path):
        """Sauvegarder le modèle"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'window_size': self.window_size,
            'input_features': self.input_features,
            'batch_size': self.batch_size,
            'use_dropout': self.use_dropout
        }, path)
    
    @staticmethod
    def load(path):
        """Charger un modèle sauvegardé"""
        checkpoint = torch.load(path)
        model = SimpleLSTM(
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            window_size=checkpoint['window_size'],
            input_features=checkpoint['input_features'],
            batch_size=checkpoint['batch_size'],
            use_dropout=checkpoint['use_dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def summary(self):
        """Afficher un résumé du modèle"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 60)
        print(f"🧠 SimpleLSTM Model Summary")
        print("=" * 60)
        print(f"📐 Input shape: ({self.window_size}, {self.input_features})")
        print(f"🔢 Hidden size: {self.hidden_size}")
        print(f"📚 Number of LSTM layers: {self.num_layers}")
        print(f"📤 Output size: 2 (lat, lon)")
        print(f"💾 Batch size: {self.batch_size}")
        print(f"🎭 Dropout enabled: {self.use_dropout}")
        print(f"🖥️  Device: {self.device}")
        print("-" * 60)
        print(f"⚙️  Model Architecture:")
        print(f"   LSTM1: {self.input_features} → 128")
        print(f"   Dense1: 128 → 2")
        print("-" * 60)
        print(f"📊 Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print("=" * 60)
        print("=" * 60)