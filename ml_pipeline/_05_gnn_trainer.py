import time
import copy
import random
from pathlib import Path

import numpy as np
import torch as tr
import matplotlib.pyplot as plt

class train:
    def __init__(self, model, data, criterion, optimizer, train_mask, test_mask,
                 device='cpu', save_location=None,
                 name=None, gradient_location=None, printinfo=False):
        self.model = model
        self.data = data
        self.y = data["bus"].y
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.device = device
        self.save_location = save_location
        self.name = name
        if gradient_location is not None:
            self.gradient_location = gradient_location
        elif save_location is not None:
            save_path = Path(save_location)
            self.gradient_location = save_path.with_name(f"{save_path.stem}_gradient.pth")
        else:
            self.gradient_location = None
            self.save_location = None
        self.last_training_state = {}
        if printinfo:
            parameters = {
                "model": model,
                "data": data,
                "criterion": criterion,
                "optimizer": optimizer,
                "device": device,
                "save_location": save_location,
                "name": name,
                "gradient_location": self.gradient_location,
            }
            for param_name, param_value in parameters.items():
                print(f"{param_name}: {param_value}")

    def _forward(self):
        return self.model(
            self.data["bus"].x,
            self.data["bus", "wire", "bus"].edge_index,
            self.data["bus", "wire", "bus"].edge_attr
        )

    def validate(self, returnResults=True, printinfo=False):
        self.model.eval()
        with tr.no_grad():
            outputs = self._forward()
            out_masked = outputs[self.test_mask].squeeze()
            y_masked = self.y[self.test_mask].squeeze()
            val_loss = self.criterion(out_masked, y_masked)
            if printinfo:
                print(f"-----Validation Loss: {val_loss.item():.4f}")
        if returnResults:
            return outputs, self.y
        else:
            return val_loss.item()

    def train_loop(self, epochs, lr_scheduler=None, save_increment=None,
                   time_limit=None, lr_decay_warm_restarts=0, clip_grad_norm=0,
                   save_best=False, predict_per_epoch=None, resume_state=None,
                   file_path=None, dist=5, penalty_multi=1):
        start_time = time.time()
        if isinstance(resume_state, (str, Path)):
            self.load_gradient(resume_state)
            resume_state = self.last_training_state
        resume_state = resume_state or {}
        running_loss = resume_state.get('running_loss', 1)
        running_penalty = 0
        loss_chart = list(resume_state.get('loss_chart', []))
        val_loss_chart = list(resume_state.get('val_loss_chart', []))
        best_val_loss = float('inf')
        best_weights = resume_state.get('best_weights', None)
        start_epoch = resume_state.get('epoch', 0)

        if predict_per_epoch is not None:
            epoch_predictions = []
            val_epoch_predictions = []

        last_completed_epoch = start_epoch
        best_checkpoint_saved = False

        for epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()

            if lr_decay_warm_restarts != 0 and hasattr(lr_scheduler, 'T_cur') and lr_scheduler.T_cur == 0 and epoch > 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    new_lr = param_group['lr'] * lr_decay_warm_restarts
                    param_group['lr'] = new_lr
                    lr_scheduler.base_lrs[i] = new_lr
                print(f"Warm restart at epoch {epoch+1}, new max learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.optimizer.zero_grad()
            outputs = self._forward()
            out_masked = outputs[self.train_mask].squeeze()
            y_masked = self.y[self.train_mask].squeeze()
            loss = self.criterion(out_masked, y_masked)
            if hasattr(self.model, 'physics_penalty'):
                total = self.model.physics_penalty["headroom"] + self.model.physics_penalty["gen_min"] + self.model.physics_penalty["flow"] + self.model.physics_penalty["nonneg"]
                loss = loss + total * penalty_multi
                running_penalty = (total.item() * 0.05 * penalty_multi + running_penalty * 0.95)
            loss.backward()

            if clip_grad_norm > 0:
                tr.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm)
            self.optimizer.step()

            loss_val = loss.item()
            running_loss = loss_val * 0.05 + running_loss * 0.95

            if epoch % 250 == 0:
                print(f"Epoch {epoch+1:>6}, Loss: {running_loss:.4f}, at {time.time() - start_time:.2f}s")
                print(f"Current penalty: {running_penalty:.4f}, rest of loss: {running_loss - running_penalty:.4f}")
                print(f"Headroom penalty: {self.model.physics_penalty['headroom']:.4f}, Gen Min penalty: {self.model.physics_penalty['gen_min']:.4f}, Flow penalty: {self.model.physics_penalty['flow']:.4f}, Non-negativity penalty: {self.model.physics_penalty['nonneg']:.4f}")
                print(f"LR at end of epoch {epoch+1}: {self.optimizer.param_groups[0]['lr']:.6f}")

            loss_chart.append(running_loss)

            epoch_val_loss = None
            if epoch % dist == 0:
                epoch_val_loss = self.validate(returnResults=False, printinfo=False)
                val_loss_chart.append(epoch_val_loss)

            if epoch % (dist*500) == 0:
                if len(val_loss_chart) > 0:
                    fig = self.plot_loss(loss_chart=loss_chart, validationTrue=True, val_loss_chart=val_loss_chart, dist=dist)
                else:
                    fig = self.plot_loss(loss_chart=loss_chart, validationTrue=False)
                if file_path is not None:
                    Path(file_path).mkdir(parents=True, exist_ok=True)
                    fig.savefig(Path(file_path) / f"loss_epoch_{epoch+1}_{self.name}.png")
                else:
                    fig.savefig(f"loss_epoch_{epoch+1}_{self.name}.png")
                plt.close(fig)

            if predict_per_epoch is not None and (epoch + 1) % predict_per_epoch == 0:
                epoch_predictions.append(self.predict())
                val_epoch_predictions.append(self.predict(mask=self.test_mask))

            improved = epoch_val_loss is not None and epoch_val_loss < best_val_loss

            if lr_scheduler is not None:
                if isinstance(lr_scheduler, tr.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = epoch_val_loss if epoch_val_loss is not None else running_loss
                    lr_scheduler.step(metric)
                else:
                    lr_scheduler.step()

            if save_best and improved:
                best_val_loss = epoch_val_loss
                self.last_training_state = {
                    'epoch': epoch + 1,
                    'running_loss': running_loss,
                    'loss_chart': list(loss_chart),
                    'val_loss_chart': list(val_loss_chart),
                    'best_val_loss': best_val_loss,
                    'best_weights': copy.deepcopy({k: v.cpu().clone() for k, v in self.model.state_dict().items()}),
                }
                if self.save_location is not None:
                    self.save_weights()
                    print(f"Saved best weights at epoch {epoch+1}")
                if self.gradient_location is not None:
                    self.save_gradient(self.gradient_location, lr_scheduler=lr_scheduler)
                    print(f"Saved best gradients at epoch {epoch+1}")
                best_checkpoint_saved = True

            if save_increment is not None and (epoch + 1) % save_increment == 0:
                self.save_weights()
                print(f"Saved weights at epoch {epoch+1}")

            if time_limit is not None and (time.time() - start_time) > time_limit:
                print(f"Time limit of {time_limit} seconds reached. Stopping training.")
                if not save_best:
                    if self.save_location is not None:
                        self.save_weights()
                    if self.gradient_location is not None:
                        self.save_gradient(self.gradient_location, lr_scheduler=lr_scheduler)
                if save_best and best_checkpoint_saved and self.save_location is not None:
                    self.load_weights(self.save_location)
                if save_best and best_checkpoint_saved and self.gradient_location is not None:
                    self.load_gradient(self.gradient_location, lr_scheduler=lr_scheduler)
                if len(val_loss_chart) > 0:
                    fig = self.plot_loss(loss_chart=loss_chart, validationTrue=True, val_loss_chart=val_loss_chart, dist=dist)
                else:
                    fig = self.plot_loss(loss_chart=loss_chart, validationTrue=False)
                if file_path is not None:
                    Path(file_path).mkdir(parents=True, exist_ok=True)
                    fig.savefig(Path(file_path) / f"loss_epoch_{epoch+1}_{self.name}.png")
                else:
                    fig.savefig(f"loss_epoch_{epoch+1}_{self.name}.png")
                plt.close(fig)
                last_completed_epoch = epoch + 1
                self.last_training_state = {
                    'epoch': last_completed_epoch,
                    'running_loss': running_loss,
                    'loss_chart': list(loss_chart),
                    'val_loss_chart': list(val_loss_chart),
                    'best_val_loss': best_val_loss,
                    'best_weights': None,
                }
                break

            last_completed_epoch = epoch + 1
            self.last_training_state = {
                'epoch': last_completed_epoch,
                'running_loss': running_loss,
                'loss_chart': list(loss_chart),
                'val_loss_chart': list(val_loss_chart),
                'best_val_loss': best_val_loss,
                'best_weights': None,
            }

        if predict_per_epoch is not None:
            return loss_chart, val_loss_chart, self.model, epoch_predictions, val_epoch_predictions
        else:
            return loss_chart, val_loss_chart, self.model

    def plot_loss(self, loss_chart, validationTrue, val_loss_chart=None, dist=1, top=2, bottom=0.9):
        plt.style.use("fivethirtyeight")
        fig, ax = plt.subplots()
        if len(loss_chart) != 0:
            if validationTrue and val_loss_chart:
                max_y = max(max(loss_chart), max(val_loss_chart)) * 1.1
            else:
                max_y = max(loss_chart) * 1.1
        else:
            max_y = 1
        min_y = min(loss_chart) * bottom
        if validationTrue and val_loss_chart:
            min_y = min(min_y, min(val_loss_chart) * bottom)
        max_y = min(val_loss_chart) * top if (validationTrue and val_loss_chart) else max_y * top
        ax.plot(range(len(loss_chart)), loss_chart, c='#ff6ff1', marker="o", label="train", linewidth=0.5, markersize=1)
        if validationTrue and val_loss_chart:
            x_vals = range(1, len(val_loss_chart) * dist, dist)
            ax.plot(x_vals, val_loss_chart, c='#8b6fff', marker="o", label="validation", linewidth=0.5, markersize=1)
        ax.set_xscale('log')
        ax.set_ylim(min_y, max_y)
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        return fig

    def save_weights(self):
        tr.save(self.model.state_dict(), self.save_location)

    def load_weights(self, path):
        self.model.load_state_dict(tr.load(path, map_location=self.device))

    def save_gradient(self, path, lr_scheduler=None):
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().clone()
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gradients': gradients,
            'training_state': copy.deepcopy(self.last_training_state),
            'torch_rng_state': tr.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if tr.cuda.is_available():
            checkpoint['cuda_rng_state_all'] = tr.cuda.get_rng_state_all()
        if lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        tr.save(checkpoint, path)

    def load_gradient(self, path, lr_scheduler=None):
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"load_gradient expected a checkpoint path as str or Path, got {type(path).__name__}."
            )
        checkpoint = tr.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for name, param in self.model.named_parameters():
            if name in checkpoint['gradients']:
                param.grad = checkpoint['gradients'][name].to(self.device)
        lr_scheduler_state = checkpoint.get('lr_scheduler_state_dict', None)
        if lr_scheduler is not None and lr_scheduler_state is not None:
            lr_scheduler.load_state_dict(lr_scheduler_state)
        if 'torch_rng_state' in checkpoint:
            tr.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state_all' in checkpoint and tr.cuda.is_available():
            tr.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
        self.last_training_state = checkpoint.get('training_state', {})
        return self.last_training_state

    def clear_plots(self):
        self.last_training_state['loss_chart'] = []
        self.last_training_state['val_loss_chart'] = []

    def print_model_stats(self, output, layerType=None):
        names = []
        means = []
        stds = []
        max_vals = []
        min_vals = []
        for name, param in self.model.named_parameters():
            if layerType is not None and layerType in name:
                param_data = param.detach().cpu()
                names.append(name)
                means.append(param_data.mean().item())
                stds.append(param_data.std().item())
                max_vals.append(param_data.max().item())
                min_vals.append(param_data.min().item())
            if layerType is None:
                param_data = param.detach().cpu()
                names.append(name)
                means.append(param_data.mean().item())
                stds.append(param_data.std().item())
                max_vals.append(param_data.max().item())
                min_vals.append(param_data.min().item())
        if output == "half":
            print("Maximum Values")
            print("-" * 85)
            max_mean = np.argmax(np.abs(means))
            max_std = np.argmax(np.abs(stds))
            max2_val = np.argmax(max_vals)
            min2_val = np.argmax(np.abs(min_vals))
            print(f'Largest Mean: {names[max_mean]}, {means[max_mean]}')
            print(f'Largest STD: {names[max_std]}, {stds[max_std]}')
            print(f'Largest Maximum: {names[max2_val]}, {max_vals[max2_val]}')
            print(f'Largest Minimum: {names[min2_val]}, {min_vals[min2_val]}')
        else:
            print(f"{'Layer Name':<35} | {'Mean':<10} | {'Std':<10} | {'Max':<10} | {'Min':<10}")
            print("-" * 85)
            for i in range(len(names)):
                print(f"{names[i]:<35} | {means[i]:10.4f} | {stds[i]:10.4f} | {max_vals[i]:10.4f} | {min_vals[i]:10.4f}")

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")

    def predict(self, mask=None):
        self.model.eval()
        with tr.no_grad():
            outputs = self._forward()
        if mask is not None:
            return outputs[mask].cpu()
        return outputs.cpu()