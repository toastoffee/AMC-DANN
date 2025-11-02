import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from metrics_utils import UpdatingAverage
from log_utils import log_info


# regular train with only one epoch
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: optim.Optimizer,
                    loss_fn: nn.Module,
                    device: torch.device,
                    header_desc: str):
    # set the model to training mode
    model.train()

    # metrics
    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()

    # start training and use tqdm as the progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (samples, labels, snr) in enumerate(dataloader):
            # convert to torch variables
            samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

            # forward
            preds: torch.Tensor = model(samples)
            loss = loss_fn(preds.float(), labels.long())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the average loss and accuracy
            loss_avg.update(loss.data)

            pred_labels = torch.argmax(preds, dim=1)
            accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
            acc_avg.update(accuracy_per_batch)

            t.set_postfix(loss='{:05.8f}'.format(loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update(1)

        train_desc = header_desc + " Train metrics, acc: {acc: .4f}, loss: {loss: .4f}".format(acc=acc_avg(),
                                                                                               loss=loss_avg())
        log_info(train_desc)

        return acc_avg, loss_avg


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             loss_fn: nn.Module,
             device: torch.device,
             header_desc: str = ""):
    model.eval()

    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()
    acc5_avg = UpdatingAverage()

    for i, (samples, labels, snr) in enumerate(dataloader):
        samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

        preds: torch.Tensor = model(samples)
        loss = loss_fn(preds.float(), labels.long())

        loss_avg.update(loss.data)

        pred_labels = torch.argmax(preds, dim=1)
        accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
        all_labels = list(range(11))
        accuracy_top_5 = top_k_accuracy_score(y_true=labels.cpu().detach().numpy(),
                                              y_score=preds.cpu().detach().numpy(),
                                              k=5, labels=all_labels)
        acc_avg.update(accuracy_per_batch)
        acc5_avg.update(accuracy_top_5)

    metric_desc = header_desc + " Eval metrics, acc: {acc: .4f},top-5:{top5: .4f} loss: {loss: .4f}".format(
        acc=acc_avg(), top5=acc5_avg(), loss=loss_avg())

    log_info(metric_desc)
    return acc_avg, loss_avg


def eval_and_get_acc(model: nn.Module,
                     dataloader: DataLoader,
                     loss_fn: nn.Module,
                     device: torch.device,
                     desc: str = ""):
    model.eval()

    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()
    acc5_avg = UpdatingAverage()

    snr_right = np.zeros(21)
    snr_all = np.zeros(21)
    snr_acc = np.zeros(21)

    for i, (samples, labels, snrs) in enumerate(dataloader):
        samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

        preds: torch.Tensor = model(samples)
        loss = loss_fn(preds.float(), labels.long())

        loss_avg.update(loss.data)

        pred_labels = torch.argmax(preds, dim=1)
        accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
        accuracy_top_5 = top_k_accuracy_score(y_true=labels.cpu().detach().numpy(),
                                              y_score=preds.cpu().detach().numpy(), k=5)

        acc_avg.update(accuracy_per_batch)
        acc5_avg.update(accuracy_top_5)

        for sample_idx in range(samples.shape[0]):
            pred = pred_labels[sample_idx]
            label = labels[sample_idx]
            snr = round(snrs[sample_idx].item())
            snr_all[snr] += 1
            if pred.item() == label.item():
                snr_right[snr] += 1

    for snr_idx in range(21):
        snr_acc[snr_idx] = snr_right[snr_idx] / snr_all[snr_idx]

    metric_desc = desc + "[evaluation complete] Eval metrics, acc: {acc: .4f},top-5:{top5: .4f} loss: {loss: .4f}".format(
        acc=acc_avg(), top5=acc5_avg(), loss=loss_avg())

    log_info(metric_desc)
    return snr_acc


def train_and_evaluate(model: nn.Module,
                       train_dataloader: DataLoader,
                       val_dataloader: DataLoader,
                       optimizer: optim.Optimizer,
                       loss_fn: nn.Module,
                       device: torch.device,
                       epochs: int,
                       model_name: str,
                       patience: int = 10,
                       min_improve_delta: float = 0.001,
                       early_stop: bool = True):
    log_info("start training: " + model_name)

    # device adaptation
    model.to(device)
    loss_fn.to(device)

    # learning rate schedulers
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # Early Stopping variables
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_triggered = False

    train_history = {
        'val_acc': [],
        'val_loss': [],
        'best_epoch': 0
    }

    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train and evaluate the model
        train_one_epoch(model, train_dataloader, optimizer, loss_fn, device,
                        f"[train] ({epoch}/{epochs})")
        acc_val, loss_val = evaluate(model, val_dataloader, loss_fn, device, "[evaluation]")

        train_history['val_acc'].append(acc_val())
        train_history['val_loss'].append(loss_val())

        if acc_val() > best_acc + min_improve_delta:
            best_acc = acc_val()
            best_epoch = epoch
            epochs_no_improve = 0
            log_info(f"[train] Epoch {epoch}: New best acc: {acc_val():.4f}, weights saved.")
            torch.save(model.state_dict(), model_name + '.pth')
            train_history['best_epoch'] = epoch
        else:
            epochs_no_improve += 1
            log_info(f"[train] Epoch {epoch}: Acc: {acc_val():.4f}, Best acc {best_acc:.4f}, "
                     f"no improvements for {epochs_no_improve} epochs")

        if early_stop and epochs_no_improve >= patience:
            log_info(f"[train] ⚠️ Early Stopping triggered! stop training on Epoch {epoch}")
            log_info(f"[train] Best_acc: {best_acc:.4f} (Epoch {best_epoch})")
            log_info(f"[train] accuracy didn't get improvements over {min_improve_delta} for {epochs_no_improve} epochs")
            early_stop_triggered = True
            break

    log_info(f"[train] ✅ train finished, actual training epochs: {epoch + 1}")
    log_info(f"[train] Best_acc: {best_acc:.4f} (Epoch {best_epoch})")
