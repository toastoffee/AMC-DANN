import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import fada_dataset
from dataset import dataset_utils
from dataset import rml_dataset
from model import fada
from train.device_utils import get_device
from model import modelutils


def train(model:        fada.FADA,
          fada_dataset: fada_dataset.FadaDataset,
          batch_size:   int,
          seed:         int,
          device:       torch.device,
          model_name:   str) -> None:

    dataset_utils.set_seeds(seed)

    epochs_1 = 30
    epochs_2 = 200
    epochs_3 = 100

    model.to(device)

    # step 1: train feature extractor and classifier -----------------------------------------------
    source_train_dataloader = DataLoader(dataset=fada_dataset.source_train_subset, batch_size=batch_size, shuffle=True)
    source_valid_dataloader = DataLoader(dataset=fada_dataset.source_valid_subset, batch_size=batch_size)
    loss_ce = torch.nn.CrossEntropyLoss()

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)
    modelutils.unfreeze(model)
    modelutils.freeze(model.DCD)

    for epoch in range(epochs_1):
        for batch_idx, (samples, labels, snr) in enumerate(source_train_dataloader):
            samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

            class_logits = model(samples)
            loss = loss_ce(class_logits.float(), labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = 0
        for batch_idx, (samples, labels, snr) in enumerate(source_valid_dataloader):
            samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

            class_logits = model(samples)
            acc += (torch.max(class_logits, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(source_valid_dataloader)), 3)

        print(f"[{model_name}]step1----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, epochs_1, accuracy))

    # step 2: train DCD -----------------------------------------------
    optimizer_D = torch.optim.Adam(model.DCD.parameters(), lr=1e-3, weight_decay=5e-3)
    modelutils.unfreeze(model)

    for epoch in range(epochs_2):
        groups, aa = fada_dataset.create_pair_groups(seed=epoch)

        n_iters = 4 * len(groups[1])    # groups[1] => samples each 4 classes
        index_list = torch.randperm(n_iters)
        mini_batch_size = 40

        loss_mean = []

        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters):
            ground_truth = index_list[index] // len(groups[1])
            x1, x2 = groups[ground_truth][index_list[index] - len(groups[1]) * ground_truth]

            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            if (index + 1) % mini_batch_size == 0:
                X1 = torch.stack(X1)    # [mini_batch_size, ..]
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)

                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_D.zero_grad()
                X_cat = torch.cat([model.encoder(X1), model.encoder(X2)], 1)
                X_cat = X_cat.detach()

                dcd_pred = model.DCD(X_cat)

                loss = loss_ce(dcd_pred, ground_truths)
                loss.backward()
                optimizer_D.step()
                loss_mean.append(loss.item())

                X1 = []
                X2 = []
                ground_truths = []

        print(f"[{model_name}]step2----Epoch %d/%d loss:%.3f" % (epoch + 1, epochs_2, np.mean(loss_mean)))

    # step 3:  -----------------------------------------------
    optimizer_g_h = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    optimizer_d = torch.optim.Adam(model.DCD.parameters(), lr=1e-3, weight_decay=5e-3)

    target_valid_dataloader = DataLoader(dataset=fada_dataset.target_valid_subset, batch_size=batch_size)

    best_acc = 0.0

    for epoch in range(epochs_3):
        # train feature_extractor and classifier, DCD is frozen
        modelutils.unfreeze(model)
        modelutils.freeze(model.DCD)

        groups, groups_y = fada_dataset.create_pair_groups(epochs_2 + epoch)
        G1, G2, G3, G4 = groups
        Y1, Y2, Y3, Y4 = groups_y
        groups_2 = [G2, G4]
        groups_y_2 = [Y2, Y4]

        n_iters = 2 * len(G2)
        index_list = torch.randperm(n_iters)

        n_iters_dcd = 4 * len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)

        mini_batch_size_g_h = 20  # data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd = 40  # data contains G1,G2,G3,G4 so use 40 as mini_batch
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels=[]

        for index in range(n_iters):

            ground_truth = index_list[index] // len(G2)
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            dcd_label = 0 if ground_truth == 0 else 2
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)

            if (index + 1) % mini_batch_size_g_h == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths_y1 = torch.LongTensor(ground_truths_y1)
                ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                dcd_labels = torch.LongTensor(dcd_labels)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths_y1 = ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels = dcd_labels.to(device)

                optimizer_g_h.zero_grad()

                encoder_X1 = model.encoder(X1)
                encoder_X2 = model.encoder(X2)

                X_cat = torch.cat([encoder_X1, encoder_X2], 1)
                y_pred_X1 = model.classifier(encoder_X1)
                y_pred_X2 = model.classifier(encoder_X2)
                y_pred_dcd = model.DCD(X_cat)

                loss_X1 = loss_ce(y_pred_X1, ground_truths_y1)
                loss_X2 = loss_ce(y_pred_X2, ground_truths_y2)
                loss_dcd = loss_ce(y_pred_dcd, dcd_labels)

                loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

                loss_sum.backward()
                optimizer_g_h.step()

                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []

        # ----training dcd ,g and h frozen
        modelutils.unfreeze(model)
        modelutils.freeze(model.feature_extractor)
        modelutils.freeze(model.classifier)
        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters_dcd):

            ground_truth = index_list_dcd[index] // len(groups[1])

            x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_d.zero_grad()
                X_cat = torch.cat([model.encoder(X1), model.encoder(X2)], 1)
                y_pred = model.DCD(X_cat.detach())
                loss = loss_ce(y_pred, ground_truths)
                loss.backward()
                optimizer_d.step()
                # loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []

        # testing
        acc = 0
        for data, labels, snr in target_valid_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = model.classifier(model.encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(target_valid_dataloader)), 3)

        print(f"[{model_name}]step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, epochs_3, accuracy))

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), model_name + '.pth')
            print(f"[{model_name}]new best accuracy reached: {best_acc}, weights saved")


if __name__ == "__main__":
    s_ds = rml_dataset.RmlHelper.rml201610a()
    t_ds = rml_dataset.RmlHelper.rml22()

    shots_selections = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    for shots in shots_selections:
        for i in range(1):
            dataset = fada_dataset.FadaDataset(s_ds, t_ds, 0.6, shots, 1)
            device: torch.device = get_device()
            model = fada.FADA()
            train(model, dataset, 512, 42, device, f"fada_weights/fada_shots-{shots}_round-{i}")
