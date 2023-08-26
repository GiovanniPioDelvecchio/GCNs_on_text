# math related libraries
import numpy as np
import torch
from sklearn.decomposition import PCA

# os related libraries
import pickle
import os 
import shutil
import os.path as osp

# task specific libraries
import stanza
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.utils import to_networkx

# libraries for visualization 
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


stanza.download("en")
nlp = stanza.Pipeline('en', use_gpu = True)

def print_example(example_sentence):
  nlp_example_sentence = nlp(example_sentence)
  to_print = []
  for sentence in nlp_example_sentence.sentences:
    cur_str = ""
    for word in sentence.words:
      if len(word.text) < 2:
        # we add an additional tabulation if the word is shorter than 2 characters
        cur_str = f"id: {word.id}\tword: {word.text}\t\t\thead id: {word.head}" 
      elif len(word.text) >= 10:
        # we remove a tabulation
        cur_str = f"id: {word.id}\tword: {word.text}\thead id: {word.head}"
      else:
        cur_str = f"id: {word.id}\tword: {word.text}\t\thead id: {word.head}"
      to_print.append(cur_str)
  for elem in to_print:
    print(elem)

# function for the construction of the dictionary containing tokens and
# dependencies
def get_tokens_and_dependencies(to_tokenize, sentence_self_loop = False, positional_links = False):
  nlp_dep = nlp(to_tokenize)
  to_return = {}
  for idx, sentence in enumerate(nlp_dep.sentences):
    token_list = np.array([])
    id_arr = np.array([], dtype=np.int32)
    heads_arr = np.array([], dtype=np.int32)
    for i, word in enumerate(sentence.words):
      if i == 0 and sentence_self_loop:
        heads_arr = np.append(heads_arr, 0)
        id_arr = np.append(id_arr, 0)
      token_list = np.append(token_list, word.text)
      heads_arr = np.append(heads_arr, word.head) 
      id_arr = np.append(id_arr, word.id) 
    if idx != 0:
        max_to_add = max(to_return[f"{idx-1} dependency"][0])
        dependency_arr = add_offset_to_links(id_arr, heads_arr, 
                                              offset = max_to_add)
    else:
        dependency_arr = add_offset_to_links(id_arr, heads_arr)
    if positional_links:
        dependency_arr = add_positional_links(dependency_arr[0], dependency_arr[1])
    to_return.update({f"{idx} tokens": token_list, f"{idx} dependency":dependency_arr})
  return to_return

def add_offset_to_links(id_arr, heads_arr, offset = 0):
    if offset == 0:
        return [id_arr, heads_arr]
    for i in range(0, len(id_arr)):
        id_arr[i] += offset + 1 
        heads_arr[i] += offset + 1 
    return [id_arr, heads_arr]


def add_positional_links(id_arr, heads_arr):
    num_nodes = id_arr.shape[0]
    for i in range(0, num_nodes):
        if i + 1 != num_nodes:
            id_arr = np.append(id_arr, id_arr[i])
            heads_arr = np.append(heads_arr, id_arr[i + 1])
            id_arr = np.append(id_arr, id_arr[i + 1])
            heads_arr = np.append(heads_arr, id_arr[i])
            
    to_return = np.vstack((id_arr, heads_arr))
    return to_return

def forest_from_token_dep_dict(tok_dep_dict, sentence_self_loop = False):
    num_sentences = len(tok_dep_dict.keys()) // 2
    all_tokens = []
    all_dep = []
    first_nodes = [0]
    for i in range(0, num_sentences):
        all_tokens.extend(tok_dep_dict[f"{i} tokens"])
        if i != 0:
            cur_dep = tok_dep_dict[f"{i} dependency"]
            if sentence_self_loop:
                first_nodes.append(cur_dep[0][0])
            else: 
                first_nodes.append(cur_dep[0][0] - 1)
            all_dep[0] = np.concatenate((all_dep[0], cur_dep[0]))
            all_dep[1] = np.concatenate((all_dep[1], cur_dep[1]))
        else:
            all_dep.extend(tok_dep_dict[f"{i} dependency"])
    return {"tokens": all_tokens, "dependency": np.array(all_dep), "first_nodes":first_nodes}

def get_forest_from_sentence(to_tokenize, sentence_self_loop = False, positional_links = True):
    return forest_from_token_dep_dict(get_tokens_and_dependencies(to_tokenize, 
                                                                  sentence_self_loop, 
                                                                  positional_links), 
                                      sentence_self_loop)

class GloveUtils:
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.vocabulary = {}
        self.embeddings_dict = {}
        self.pca = PCA(n_components=3)
        self.max_proj = 0
        self.min_proj = 0
        
        with open(glove_path, 'r', encoding="utf-8") as f:
          for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            self.embeddings_dict[word] = vector
            
    def embed_to_GloVe(self, tokens_to_embed, pca_flag = False):
      # Initialize an empty list to store the embeddings
      embeddings = []

      # Iterate through the tokens and retrieve the embeddings
      for token in tokens_to_embed:
          if token in self.embeddings_dict:
              embeddings.append(self.embeddings_dict[token])
              self.vocabulary.update({token:self.embeddings_dict[token]})
          else:
            # If the token is not found in the vocabulary, you can assign a random embedding or any other handling strategy
            random_embed = np.random.uniform(-0.25, 0.25, 50)
            embeddings.append(random_embed)
            self.vocabulary.update({token:random_embed})

      # Convert the list of embeddings to a NumPy array
      if pca_flag:
          self.__fit_pca__()
      embeddings = np.array(embeddings)
      return embeddings
    
    def __fit_pca__(self):
        to_fit = [self.vocabulary[word] for word in list(self.vocabulary.keys())]
        fitted_vocab = self.pca.fit_transform(to_fit)
        self.max_proj = np.amax(fitted_vocab)
        self.min_proj = np.amin(fitted_vocab)
    
    def project_embedding(self, embed_to_project):
        #self.__fit_pca__()
        projected_embed = self.pca.transform(np.array([embed_to_project]))
        to_return = (projected_embed - self.min_proj) / (self.max_proj - self.min_proj)
        return to_return
    
    def project_embeddings(self, embeddings):
        projected_embed = self.pca.transform(embeddings)
        to_return = (projected_embed - self.min_proj) / (self.max_proj - self.min_proj)
        return to_return
    
    def project_tokens(self, tokens_to_project):
        n_tok = len(tokens_to_project)
        embeddings = self.embed_to_GloVe(tokens_to_project)
        to_return = np.zeros((n_tok, 3))
        for i in range(0, n_tok):
            to_return[i, :] = self.project_embedding(embeddings[i, :])
        return to_return
    
    def serialize_vocab(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.vocabulary, f)
        
    def load_vocab(self, path, pca_flag = False):
        with open(path, 'rb') as f:
            self.vocabulary = pickle.load(f)
        if pca_flag:
           self.__fit_pca__()

glove_path = 'content/embed/glove.6B.50d.txt'
g_utils = GloveUtils(glove_path)

# class containing the graph data that must be fed into the GCNs or GAT networks
class Dataset_from_sentences(Dataset):
    def __init__(self, name, path_were_save, drive_dir, sentences_list, y_values, transform=None):
      self.name = name
      self.drive_dir = drive_dir
      self.root = path_were_save
      self.raw_url = str(self.drive_dir + self.name + ".pt")
      print(self.raw_paths)
      self.data_list = []
      self.sentences_list = sentences_list
      self.y_values = y_values
      if os.path.exists(self.raw_paths[0]):
        self.data_list = torch.load(self.raw_paths[0])
        self.data_list.x = self.data_list.x.type(torch.FloatTensor)
        self._indices = range(self.len())
        self.transform = None
      else:
        print("something strange happened")
        super().__init__(path_were_save, transform)
        

    @property
    def processed_file_names(self):
      return self.raw_paths[0]

    @property 
    def raw_paths(self):
      to_return = self.root + "/" + self.name + ".pt"
      return [to_return]
      

    def download(self):
      if os.path.exists(self.raw_url):
        shutil.copy(self.raw_url, self.raw_paths[0])
      else:
        pass


    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def __build_graph_Data_with_GloVe__(self, sentence, y_val, return_dep_dict = False, pca_flag = False):  
        tok_dep_dict = get_forest_from_sentence(sentence)
        intra_sentence_data_list = []

        if len(tok_dep_dict.keys()) > 0:
            glove_embeddings = torch.tensor(g_utils.embed_to_GloVe(tok_dep_dict["tokens"], pca_flag))
            glove_sentence_avg = torch.mean(glove_embeddings, dim=0)
            # the sentence is modelled as the directed dependency graph where the nodes
            # have the bert embeddings as features, since the verb of the main
            # sentence points to 0, node 0 has the pooler output as its features,
            # as they represent the meaning of the whole sentence 
            node_features = torch.vstack((glove_sentence_avg, glove_embeddings)) 
            reshaped = glove_sentence_avg.reshape(1, list(glove_sentence_avg.shape)[0])
            for first_node in tok_dep_dict["first_nodes"]:
                if first_node != 0:
                    node_features = torch.cat((node_features[0:first_node, :],
                              reshaped, node_features[first_node:, :]), 0)

            edge_idxs = torch.tensor(tok_dep_dict[f"dependency"], dtype = torch.int64)
            data = Data(x = node_features, edge_index = edge_idxs, y = y_val)
            intra_sentence_data_list.append(data)
            batch = Batch.from_data_list(intra_sentence_data_list)
            if return_dep_dict:
                return batch, tok_dep_dict
        return batch
    

    def to(self, device):
      self.data_list.to(device)
      return self

    def process(self):
      num_invalid = 0
      for idx, elem in tqdm(enumerate(self.sentences_list)):
        #dataset_name = f"data_{idx - num_invalid}.pt"
        to_save = self.__build_graph_Data_with_GloVe__(elem, self.y_values[idx])
        if to_save is not None:
          self.data_list.append(to_save)
        else:
          num_invalid += 1
      torch.save(Batch.from_data_list(self.data_list), self.raw_paths[0])
      torch.save(Batch.from_data_list(self.data_list), self.raw_url)


# function needed to clear a directory, it is necessary to call this function
# if some modifications to the dataset must be performed
def delete_processed_files(directory):
  if os.path.exists(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
              for elem in os.listdir(file_path):
                os.unlink(osp.join(file_path, elem))
              os.rmdir(file_path)
        except Exception as e:
            print('Cannot eliminate {}: {}'.format(file_path, e))
    print(f"Now {directory} is empty")
  else:
      print("The directory does not exist")

def visualize_graph(G, td_dict = []):
    to_draw = to_networkx(G, to_undirected = False)
    embeds = G["x"]
    embeds = embeds.cpu()
    n_nodes = embeds.shape[0]
    colors = np.power(g_utils.project_embeddings(embeds), 2)
    color = "red"
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    if td_dict == []:
        nx.draw_networkx(to_draw, pos=nx.spring_layout(to_draw, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    else:
        labels = {}
        j = 0
        for i in range(0, n_nodes):
            if i in td_dict["first_nodes"]:
                labels.update({i:"Sentence"})
            else:
                current_token = td_dict["tokens"][j]
                labels.update({i:current_token})
                j += 1
        nx.draw_networkx(to_draw, pos=nx.spring_layout(to_draw, seed=42), with_labels=True,
                         labels = labels,
                         node_color=colors, cmap="Set2")
    plt.show()
    
def visualize_hidden_graph(x_features, links, labels = [], project_flag = True, k_custom = None):
    to_draw = nx.MultiDiGraph()
    links_for_the_net = list(zip(links[0, :].detach().cpu().numpy(), 
                                 links[1, :].detach().cpu().numpy()))
    to_draw.add_edges_from(links_for_the_net)
    embeds = x_features.detach().cpu()
    n_nodes = embeds.shape[0]
    if project_flag:
        colors = np.power(g_utils.project_embeddings(embeds), 2)
    else:
        colors = embeds # if embeds has a number of columns different from 3, some
                        # aggregation must be performed before calling this
                        # function with project_flag == False
    
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    if labels == []:
        nx.draw_networkx(to_draw, pos=nx.spring_layout(to_draw, seed=42, k = k_custom), with_labels=False,
                         node_color=colors, cmap="Set2")
    else:
        labels = dict(zip(np.arange(1, n_nodes), labels))
        labels.update({0:"Sentence"})
        nx.draw_networkx(to_draw, pos=nx.spring_layout(to_draw, seed=42, k = k_custom), with_labels=True,
                         labels = labels,
                         node_color=colors, cmap="Set2")
    plt.show()

print("Utils have been correctly loaded")