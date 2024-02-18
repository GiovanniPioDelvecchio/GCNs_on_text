import utils
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import save, load
import torch.nn as nn
from torchsummary import summary

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.utils import scatter
from torch_geometric.data import download_url

from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear, Dropout

from functools import partial
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import PolynomialLR
from torch_geometric.nn.pool import global_max_pool
from torch.optim.lr_scheduler import OneCycleLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


tweet_list_train = []
sentiment_list_train = []
tweet_list_val = []
sentiment_list_val = []
tweet_list_test = []
sentiment_list_test = []
cv19_graph_data_train = utils.Dataset_from_sentences("train", "GraphDataset/train/",
                                                     "/content/drive/MyDrive/glove_twitter_100_unnormalized/train/",
                                                     tweet_list_train, sentiment_list_train)
cv19_graph_data_val = utils.Dataset_from_sentences("val", "GraphDataset/val/",
                                                     "/content/drive/MyDrive/glove_twitter_100_unnormalized/val/",
                                                     tweet_list_train, sentiment_list_train)
cv19_graph_data_test = utils.Dataset_from_sentences("test", "GraphDataset/test/",
                                                    "/content/drive/MyDrive/glove_twitter_100_unnormalized/test/",
                                                    tweet_list_test, sentiment_list_test)

print("Loaded training dataset:")
print(cv19_graph_data_train)
print("Loaded val dataset:")
print(cv19_graph_data_val)
print("Loaded test dataset:")
print(cv19_graph_data_test)

# actual GAT class
class GAT_lstm(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, n_filters, dim_out, heads= 8):
    super().__init__()
    # dim_in is the number of node features, dim_h is the dimension
    # of the hidden layer, dim_out is the dimension of the output
    # feature vector


    #self.first_linear = Linear(dim_in, dim_in)

    self.gat_list_1 = torch.nn.ModuleList([GATv2Conv(dim_in, dim_in // heads, heads = heads)
                                         for i in range(0, n_filters)])
    self.gat_list_2 = torch.nn.ModuleList([GATv2Conv(dim_in, dim_in // heads, heads = heads)
                                         for i in range(0, n_filters)])
    self.gat_list_3 = torch.nn.ModuleList([GATv2Conv(dim_in, dim_in // heads, heads = heads)
                                         for i in range(0, n_filters)])

    self.l_list = torch.nn.ModuleList([Linear(dim_in, 1)
                                      for i in range(0, n_filters)])

    self.h0 = torch.randn(1, dim_in)
    self.c0 = torch.randn(1, dim_in)
    self.h0 = self.h0.to(device)
    self.c0 = self.c0.to(device)
    self.lstm = nn.LSTM(dim_in, dim_in, num_layers = 1, dropout = 0, bidirectional = False)
    self.classifier = Linear(n_filters, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=1e-3,
                                      weight_decay=5e-5)

  def forward(self, x, edge_index, batch, enable_log = False):
    # the parameters of the forward correspond to data.x and data.edge_index
    # where data is a Data object like those described above;
    stack_list = []
    batch_elements = batch.unique()
    for elem in batch_elements:
      idxs = torch.where(batch == elem)
      current_graph = x[idxs]
      self.lstm.flatten_parameters()
      to_stack, (hn, cn) = self.lstm(current_graph, (self.h0, self.c0))
      stack_list.append(to_stack)

    h_to_gat = torch.vstack(stack_list)
    #print(h_to_gat.shape)
    #print(x.shape)

    #h_stacked = global_mean_pool(h_stacked, batch)
    #h_stacked = self.classifier(h_stacked)

    h_list = []
    for i, gat_l in enumerate(self.gat_list_1):
        h = h_to_gat + gat_l(h_to_gat, edge_index)
        h = h.tanh()
        h = h + self.gat_list_2[i](h, edge_index)
        h = h.tanh()
        h = h + self.gat_list_3[i](h, edge_index)
        if enable_log:
            print("h shape: " + str(h.shape))
            utils.visualize_hidden_graph(h, edge_index)
        h = global_mean_pool(h, batch)
        h = self.l_list[i](h)
        h_list.append(h)

    h_layers = torch.hstack(h_list)
    h_layers = self.classifier(h_layers)
    return h_layers
  
class EarlyStopper:
  def __init__(self, patience=1, min_delta=0):
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0
      self.min_validation_loss = np.inf

  def early_stop(self, validation_loss):
      if validation_loss < self.min_validation_loss:
          self.min_validation_loss = validation_loss
          self.counter = 0
      elif validation_loss > (self.min_validation_loss + self.min_delta):
          self.counter += 1
          if self.counter >= self.patience:
              return True
      return False

def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation losses over the course of training.
    Args:
        train_losses: A list of training losses.
        val_losses: A list of validation losses.
    """
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

import time
def train(model, strat_train, strat_val, partial_scheduler, epochs = 30, batch_size = 30, print_every = 1, path='Model'):
    """Train a GNN model and return the trained model."""
    batch_size = batch_size
    criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.01)
    optimizer = model.optimizer
    scheduler = partial_scheduler(optimizer)
    print(type(scheduler))
    loader_train =  DataLoader(strat_train.data_list, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(strat_val.data_list, batch_size=batch_size, shuffle=True)
    model.train()
    early_stopper = EarlyStopper(patience=10, min_delta=0.2)

    train_losses = []
    val_losses = []

    best_model = []
    best_acc_val = 0
    epochs_to_return = 0
    start_t = time.time()
    #scheduler.step()
    for epoch in range(epochs+1):
      mean_loss_train = 0
      mean_acc_train = 0
      for i, batch in enumerate(loader_train):
        # Training

        out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
        loss_train = criterion(out, batch.y.long().to(device))
        mean_loss_train += loss_train.item()

        acc_train = accuracy(out.argmax(dim=1), batch.y.to(device))
        mean_acc_train += acc_train
        loss_train.backward()
        with torch.no_grad():
            optimizer.step()
            if type(scheduler) == OneCycleLR:
                scheduler.step()
            optimizer.zero_grad()
      print(i)
      mean_loss_train /= (i + 1)
      mean_acc_train /= (i + 1)
      train_losses.append(mean_loss_train)
      if type(scheduler) != OneCycleLR:
          scheduler.step()

      mean_loss_val = 0
      mean_acc_val = 0
      #model.eval()
      with torch.no_grad():
          for i, batch in enumerate(loader_val):
            out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
            loss_val = criterion(out, batch.y.long().to(device))
            mean_loss_val += loss_val.item()
            acc_val = accuracy(out.argmax(dim=1), batch.y.to(device))
            mean_acc_val += acc_val
            #visualize_embedding(embed, batch.y, epoch, loss_val)
          mean_loss_val /= (i + 1)
          mean_acc_val /= (i + 1)
          val_losses.append(mean_loss_val)
          if mean_acc_val > best_acc_val:
                best_acc_val = mean_acc_val
                best_model_state_dict = model.state_dict
                torch.save(model.state_dict(), path + '_ckpt')
                epochs_to_return = epoch

      if(epoch % print_every == 0):
        print(f'Epoch {epoch:>3} | Train Loss: {mean_loss_train:.3f} | Train Acc: '
              f'{mean_acc_train*100:>6.2f}%')
        print(f'Epoch {epoch:>3} | Val Loss: {mean_loss_val:.3f} | Val Acc: '
              f'{mean_acc_val*100:>6.2f}%')
        print("learning rate: " + str(scheduler.get_last_lr()))
        print("elapsed time: %.2f" % (time.time() - start_t))
        start_t = time.time()
      if early_stopper.early_stop(mean_loss_val):
        print("early stopping was triggered, final loss:" + str(mean_loss_val))
        break
    plot_losses(train_losses, val_losses)
    return best_model_state_dict, best_acc_val, epochs_to_return

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def build_scheduler_list(epochs = 30, max_lr = 1e-3, num_batches = 1):
    schedulers = []

    #schedulers += [partial(PolynomialLR,
    #                     total_iters = epochs, # The number of steps that the scheduler decays the learning rate.
    #                     power = i) for i in range(2, 3)] # The power of the polynomial. # 2 to 1
    schedulers += [partial(StepLR,
                                  step_size=epochs//i, gamma=0.5) for i in range(3, 4)] # for i in range(3, 5)
    schedulers += [partial(CosineAnnealingLR,
                              T_max = epochs, # Maximum number of iterations.
                              eta_min = min_lr) for min_lr in [1e-8]] # Minimum learning rate, tested values were 1e-6, 1e-7, 1e-8
    #schedulers += [partial(OneCycleLR,
    #                   max_lr = max_lr, # Upper learning rate boundaries in the cycle for each parameter group
    #                   steps_per_epoch = num_batches, # The number of steps per epoch to train for.
    #                   epochs = epochs, # The number of epochs to train for.
    #                   anneal_strategy = 'cos')] # Specifies the annealing strategy
    return schedulers

def hyperparameter_tuning(model, strat_train, strat_val, scheduler_list, batch_size = 30, epochs = 30):
    print(f"batch size: {batch_size}")
    best_valid_acc = 0
    best_model = []
    best_hyper_params = []
    best_val_acc = 0
    i = 1
    for partial_scheduler in scheduler_list:
        #path = f'/content/drive/MyDrive/graphmod/GAT_best_{partial_scheduler.func.__name__}_{i}'
        path = f'./GAT_best_{partial_scheduler.func.__name__}_{i}'
        model_out, mean_acc_val, epoch = train(copy.deepcopy(model), strat_train, strat_val,
                                               partial_scheduler, epochs, batch_size = batch_size, print_every=1, path=path)
        torch.save(model_out, path)
        if mean_acc_val > best_val_acc:
            best_val_acc = mean_acc_val
            best_model = model_out
            best_hyper_params = [partial_scheduler, epoch]
            print(f"Improved result: acc {best_val_acc:.3f}, scheduler:\n {partial_scheduler}\nepoch: {epoch}")
        i = i + 1
    return best_hyper_params, best_model


model_gat = GAT_lstm(cv19_graph_data_train.num_node_features,
                cv19_graph_data_train.num_classes*2,
                cv19_graph_data_train.num_classes, heads = 10).to(device)
print(model_gat)

epochs = 300
# da rifare esperimenti col polinomial 2
# training information from stepLR (2) to CosineAnnealing
print(get_lr(model_gat.optimizer))
scheduler_list = build_scheduler_list(epochs=epochs,
                                      num_batches=(len(cv19_graph_data_train.data_list)//256) + 4)
print(scheduler_list)
trained_gat = hyperparameter_tuning(model_gat, cv19_graph_data_train,
                                    cv19_graph_data_val, scheduler_list,
                                    batch_size = 256, epochs = epochs)

def eval_model(model, loader_val, criterion, batch_size):
  mean_loss_val = 0
  mean_acc_val = 0
  #model.eval()
  idx_wrong_samples = np.array([])
  with torch.no_grad():
      for i, batch in tqdm(enumerate(loader_val)):
        out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
        loss_val = criterion(out, batch.y.long().to(device))
        mean_loss_val += loss_val.item()
        acc_val = accuracy(out.argmax(dim=1), batch.y.to(device))
        mean_acc_val += acc_val
        curr_idx_wrong_samples = (batch_size * i) + np.flatnonzero(out.argmax(dim=1).cpu() != batch.y)
        idx_wrong_samples = np.concatenate((idx_wrong_samples, curr_idx_wrong_samples))
        #visualize_embedding(embed, batch.y, epoch, loss_val)
      mean_loss_val /= (i + 1)
      mean_acc_val /= (i + 1)
  return mean_loss_val, mean_acc_val, idx_wrong_samples

try:
    trained_gat_model = trained_gat[1]
except Exception as e:
    saved_dict = torch.load('GAT_best_CosineAnnealingLR_2_ckpt')
    print(saved_dict['loss'])
    model_gat.load_state_dict(saved_dict["model_state_dict"])
    trained_gat_model = model_gat

print(trained_gat_model)

batch_size = 64
loader_test = DataLoader(cv19_graph_data_test.data_list, batch_size=batch_size, shuffle=False)
criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.01)
mean_loss_test, mean_acc_test, idx_wrong_samples = eval_model(trained_gat_model, loader_test, criterion, batch_size)

print(f'Test loss: {mean_loss_test}')
print(f'Test accuracy: {mean_acc_test}')

print(idx_wrong_samples.shape)
len(cv19_graph_data_test)