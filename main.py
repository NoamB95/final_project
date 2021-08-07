import datetime
from torch.utils.data import DataLoader
from mean_teacher import architectures, data
from mean_teacher import optim_weight_swa
from mean_teacher.data import NO_LABEL
import torchvision
import time
import os
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import MNIST, CIFAR100, FashionMNIST
from torch.utils.data import ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from pycm import *


def create_model(arch, pretrained, num_classes, no_grad=False):
    model_factory = architectures.__dict__[arch]
    model_params = dict(pretrained=pretrained, num_classes=num_classes)
    model = model_factory(**model_params)
    model = nn.DataParallel(model)
    if no_grad:
        for param in model.parameters():
            param.detach_()
    return model


def update_batchnorm(model, train_loader):
    model.train()
    for i, ((input, ema_input), target) in enumerate(train_loader):
        if i > 100:
            return
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        model(input_var)


def dict_values_avg(d):
    vals = list(d.values())
    s = 0
    for i in vals:
        if isinstance(i, float):
            s += i
    return s/len(vals)


def compute_auc_prcurve(class_pre, class_re):
    aucpr_avg = 0.0
    for p, r in zip(class_pre, class_re):
        if isinstance(p, float) and isinstance(r, float):
            aucpr_avg += auc(p, r)
    return aucpr_avg/len(class_pre)


def run_on_dataset(dataset, num_of_classes, set, net, res, part):
    kfold = KFold(n_splits=k_folds, shuffle=True)

    results = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        print(f'FOLD {fold}')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler,
                                                  num_workers=num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler,
                                                 num_workers=num_workers, pin_memory=True, shuffle=False,
                                                 drop_last=False)

        if net is 'fastswa':
            network = create_model('cifar_cnn', False, num_of_classes)
            fastswa_net_optim = optim_weight_swa.WeightSWA(network)
        elif net is 'resnet18':
            network = torchvision.models.resnet18(pretrained=True)
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, num_of_classes)

        network.to(device)

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        train_start_time = time.time()

        with torch.no_grad():

            for epoch in range(0, num_epochs):

                print(f'Starting epoch {epoch+1}')
                print(datetime.datetime.now())

                if net is 'fastswa' and epoch % interval == 0:
                    fastswa_net_optim.update(network)
                    update_batchnorm(network, trainloader)

                current_loss = 0.0

                for i, ((inputs, inputs2), targets) in enumerate(trainloader, 0):

                    inputs, targets = inputs.to(device), targets.to(device)

                    inputs_var = torch.autograd.Variable(inputs)
                    if torch.cuda.is_available():
                        targets_var = torch.autograd.Variable(targets.cuda(non_blocking=True))
                    else:
                        targets_var = torch.autograd.Variable(targets)

                    optimizer.zero_grad()

                    outputs = network(inputs_var)
                    if net is 'fastswa':
                        outputs = outputs[0]

                    loss = torch.autograd.Variable(loss_function(outputs, targets_var), requires_grad=True)

                    loss.backward()

                    optimizer.step()

                    current_loss += loss.item()
                    if i % 500 == 499:
                        print('Loss after mini-batch %5d: %.3f' %
                              (i + 1, current_loss / 500))
                        current_loss = 0.0

                print(f'finished epoch {epoch+1} for data {set} in model {net} part {part}')
                print(datetime.datetime.now())

        train_end_time = time.time()
        training_time = train_end_time - train_start_time

        # Process is complete.
        print(f'Training process has finished for data {set} in model {net} part {part}. Saving trained model.')
        print(datetime.datetime.now())

        # Print about testing
        print('Starting testing')

        # # Saving the model
        # save_path = f'./model-fold-{fold}-{set}-{part}-{net}.pth'
        # torch.save(network.state_dict(), save_path)

        correct, total = 0, 0
        all_pred, all_targets = list(), list()
        avg_inf_time = 0
        with torch.no_grad():

            for i, ((inputs, inputs2), targets) in enumerate(testloader, 0):

                inference_time_curr = time.time()

                inputs, targets = inputs.to(device), targets.to(device)

                inputs_var = torch.autograd.Variable(inputs)
                if torch.cuda.is_available():
                    targets_var = torch.autograd.Variable(targets.cuda(non_blocking=True))
                else:
                    targets_var = torch.autograd.Variable(targets)

                outputs = network(inputs_var)
                if net is 'fastswa':
                    outputs = outputs[0]

                _, predicted = torch.max(outputs.data, 1)
                total += targets_var.size(0)
                correct += (predicted == targets_var).sum().item()

                all_pred += predicted.tolist()
                all_targets += targets_var.tolist()

                if i*len(inputs) % 1000 == 0:
                    t = time.time() - inference_time_curr
                    avg_inf_time += t
                    if avg_inf_time != 0:
                        avg_inf_time = avg_inf_time / 2

        cm = ConfusionMatrix(actual_vector=all_targets, predict_vector=all_pred)
        class_precision = cm.PPV
        class_recall = cm.TPR
        tpr = dict_values_avg(class_recall)
        fpr = dict_values_avg(cm.FPR)
        precision = dict_values_avg(class_precision)
        roc_auc = dict_values_avg(cm.AUC)
        accuracy = 100.0 * correct / total
        auc_prcurve = compute_auc_prcurve(class_precision, class_recall)
        data_to_write = [set+str(part+1), net, fold, network.parameters(), accuracy, tpr, fpr, precision, roc_auc,
                         auc_prcurve, training_time, avg_inf_time]
        a_series = pd.Series(data_to_write, index=res.columns)
        res = res.append(a_series, ignore_index=True)

        print('--------------------------------')
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f'Finished for data {set} in model {net} part {part}.')
    print(datetime.datetime.now())
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    acc_sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        acc_sum += value
    print(f'Average: {acc_sum/len(results.items())} %')
    return res


def get_transformation():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    return transformation


def get_data(set, part):
    transformation = get_transformation()

    if set == 'MNIST':
        dataset_train_part = MNIST(os.getcwd(), download=True, transform=transformation, train=True)
        dataset_test_part = MNIST(os.getcwd(), download=True, transform=transformation, train=False)

    elif set == 'CIFAR100':
        dataset_train_part = CIFAR100(os.getcwd(), download=True, transform=transformation, train=True)
        dataset_test_part = CIFAR100(os.getcwd(), download=True, transform=transformation, train=False)

    elif set == 'FashionMNIST':
        dataset_train_part = FashionMNIST(os.getcwd(), download=True, transform=transformation, train=True)
        dataset_test_part = FashionMNIST(os.getcwd(), download=True, transform=transformation, train=False)

    train_len = len(dataset_train_part)
    test_len = len(dataset_test_part)
    train_div = int(train_len/dataset_divide)
    test_div = int(test_len/dataset_divide)
    num_of_classes = len(dataset_train_part.classes)
    dataset_train_part = torch.utils.data.Subset(dataset_train_part, list(range(train_div*part, train_div*(part+1))))
    dataset_test_part = torch.utils.data.Subset(dataset_test_part, list(range(test_div*part, test_div*(part+1))))
    dataset = ConcatDataset([dataset_train_part, dataset_test_part])
    return dataset, num_of_classes


if __name__ == '__main__':
    columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation Fold', 'Hyper-Parameters', 'Accuracy', 'TPR', 'FPR',
               'Precision', 'AUC', 'PR-Curve', 'Training Time - Seconds', 'Inference Time - Seconds']
    results = pd.DataFrame(columns=columns)

    k_folds = 10
    num_epochs = 50
    interval = 4
    dataset_divide = 5
    num_workers = 1
    batch_size = 10
    loss_function = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    big_datasets = ['MNIST', 'FashioMNIST', 'CIFAR100']
    models = ['resnet18', 'fastswa']

    device = "cpu"
    if torch.cuda.is_available():
        print('Running on GPU')
        device = "cuda:0"
        loss_function = nn.CrossEntropyLoss().cuda()
        num_workers = 2

    print(device)

    for model in models:
        for set in big_datasets:
            for part in range(dataset_divide):
                ds, num_of_classes = get_data(set, part)
                results = run_on_dataset(ds, num_of_classes, set, model, results, part)

    results.to_excel('results-{}.xlsx'.format(time.strftime("%Y%m%d-%H%M%S")))

