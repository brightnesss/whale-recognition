import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
from tqdm import tqdm
import os

from TripletDataset import TripletDataset
from ResNet import TripletsModel
import utils
from utils import PairwiseDistance
from eval_metrix import evaluate
from tensorboardX import SummaryWriter


class TripletMarginLoss(Function):
    """Triplet loss function.
    """

    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


l2_dist = PairwiseDistance(2)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize])

train_dir = TripletDataset(data_dir='/data1/whale/train', n_triplets=1000000, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dir,
                                           batch_size=12, shuffle=False)

model_dir = '/data1/model-zhanghan/whale'


def main():
    # instantiate model and initialize weights
    model = TripletsModel(embedding_size=256,
                          num_classes=4251,
                          pretrained=True)

    model.cuda()

    optimizer = create_optimizer(model, 0.01)

    summary = SummaryWriter()

    for epoch in range(100):
        print('start epoch [{}/{}]'.format(epoch + 1, 100))
        train(train_loader, model, optimizer, epoch, summary)


def train(train_loader, model, optimizer, epoch, summary: SummaryWriter()):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))
    labels, distances = [], []

    margin = 1.0

    for batch_idx, (anchor, positive, negative, label_p, label_n) in pbar:

        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor, positive, negative = Variable(anchor), Variable(positive), \
                                     Variable(negative)

        # compute output
        out_a, out_p, out_n = model(anchor), model(positive), model(negative)

        # Choose the hard negatives
        d_p = l2_dist.forward(out_a, out_p)
        d_n = l2_dist.forward(out_a, out_n)
        all = (d_n - d_p < margin).cpu().data.numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue

        out_selected_a = Variable(torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).cuda())
        out_selected_p = Variable(torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).cuda())
        out_selected_n = Variable(torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).cuda())

        selected_anchor = Variable(torch.from_numpy(anchor.cpu().data.numpy()[hard_triplets]).cuda())
        selected_positive = Variable(torch.from_numpy(positive.cpu().data.numpy()[hard_triplets]).cuda())
        selected_negative = Variable(torch.from_numpy(negative.cpu().data.numpy()[hard_triplets]).cuda())

        selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
        selected_label_n = torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
        triplet_loss = TripletMarginLoss(margin).forward(out_selected_a, out_selected_p, out_selected_n)

        class_anchor = model.forward_classifier(selected_anchor)
        class_positive = model.forward_classifier(selected_positive)
        class_negative = model.forward_classifier(selected_negative)

        criterion = nn.CrossEntropyLoss()
        predicted_labels = torch.cat([class_anchor, class_positive, class_negative])
        true_labels = torch.cat(
            [Variable(selected_label_p.cuda()), Variable(selected_label_p.cuda()), Variable(selected_label_n.cuda())])

        cross_entropy_loss = criterion(predicted_labels.cuda(), true_labels.cuda())

        loss = cross_entropy_loss + triplet_loss
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        if epoch == 30 or epoch == 60:
            adjust_lr(optimizer, 0.1)

        # log loss value
        summary.add_scalar('triplet_loss', triplet_loss.data[0], epoch + 1)
        summary.add_scalar('cross_entropy_loss', cross_entropy_loss.data[0], epoch + 1)
        summary.add_scalar('total_loss', loss.data[0], epoch + 1)
        if batch_idx % 10 == 0:  # log data every 10 batch
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t # of Selected Triplets: {}'.format(
                    epoch, batch_idx * len(anchor), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data[0], len(hard_triplets[0])))

        dists = l2_dist.forward(out_selected_a,
                                out_selected_n)  # torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(np.zeros(dists.size(0)))

        dists = l2_dist.forward(out_selected_a,
                                out_selected_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(dists.size(0)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist[0] for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
    print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    summary.add_scalar('Train Accuracy', np.mean(accuracy), epoch + 1)

    plot_roc(fpr, tpr, figure_name="roc_train_epoch_{}.png".format(epoch))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(model_dir, epoch))


def create_optimizer(model, lr):
    optimizer = optim.SGD([
        {'params': model.pretrain.parameters(), 'lr': lr * 0.1},
        {'params': model.fc.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=lr, momentum=0.9)
    return optimizer


def adjust_lr(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
    return optimizer


def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(model_dir, figure_name), dpi=fig.dpi)


if __name__ == '__main__':
    main()
