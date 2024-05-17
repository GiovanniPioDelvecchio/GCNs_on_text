""" 
-------------------------------------------------------------------------- 
NLP project work
Summary: Casting text classification to Graph Classification for Sentiment Analysis of Tweets
Members:

-Dell'Olio Domenico
-Delvecchio Giovanni Pio
-Disabato Raffaele

The project was developed in order to evaluate the effectiveness of Graph Neural network on a sentiment analysis task proposed in the challenge:
https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?resource=download

We decided to implement and test various architectures, including commonly employed transformer-based architectures, in order to compare their performances.
These architectures were either already present at the state of the art or were obtained as a result of experiments.
-------------------------------------------------------------------------- 
This script contains:
- Helper functions and classes to perform dataset conversion to graph structures with both bert and glove embeddings
- Helper functions for graph visualization
"""

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
from torch.nn.functional import normalize
from transformers import BertTokenizerFast, BertModel

# libraries for visualization
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


## Loads stanza nlp pipeline
# disable the commented kv pairs to get the appropriate models
# for the pipeline
config = {
          'processors': 'tokenize,lemma,pos,depparse,ner',
          'lang': 'en',
          #'tokenize_pretokenized': True, # disable tokenization
          #'tokenize_model_path': '../TweebankNLP/twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt',
          #'lemma_model_path': '../TweebankNLP/twitter-stanza/saved_models/lemma/en_tweet_lemmatizer.pt',
          #"pos_model_path": '../TweebankNLP/twitter-stanza/saved_models/pos/en_tweet_tagger.pt',
          #"depparse_model_path": '../TweebankNLP/twitter-stanza/saved_models/depparse/en_tweet_parser.pt',
          #"ner_model_path": '../TweebankNLP/twitter-stanza/saved_models/ner/en_tweet_nertagger.pt',
          "use_gpu": True,
}

stanza.download("en")
nlp = stanza.Pipeline(**config)


def get_tokens_and_dependencies(to_tokenize, sentence_self_loop=False, positional_links=False):
  """
  This function constructs a dictionary containing tokens and their dependencies from a given sentence.

  :param to_tokenize: The string containing the sentence to be tokenized and analyzed.
  :param sentence_self_loop: A boolean indicating whether the graph has the out-edge from the sentence-representing note entering in the node itself.
  :param positional_links: A boolean indicating whether to add edges between tokens based
                        on their relative position in the sentence (e.g., previous word, next word).
  :return: A dictionary containing two key-value pairs for each sentence in the input string:
        * "{idx} tokens": A NumPy array containing the tokens of the sentence at index idx.
        * "{idx} dependency": A NumPy array representing the dependency links in the sentence at index idx.
  """
  # tokenize sentence
  nlp_dep = nlp(to_tokenize)
  to_return = {}
  for idx, sentence in enumerate(nlp_dep.sentences):
    token_list = np.array([])
    id_arr = np.array([], dtype=np.int32)
    heads_arr = np.array([], dtype=np.int32)
    #for each word
    for i, word in enumerate(sentence.words):
      # adds edges
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

def add_offset_to_links(id_arr, heads_arr, offset=0):
    """
    This function adds a specified offset value to the indices in two NumPy arrays representing dependency links.

    :param id_arr: A NumPy array containing the ids of tokens.
    :param heads_arr: A NumPy array containing the head token indices.
    :param offset: The offset value to be added to each index in the arrays. Defaults to 0.

    :return: (list) A list containing two NumPy arrays, the first being the id array with offset,
                and the second being the head token array with offset.
    """
    if offset == 0:
        return [id_arr, heads_arr]
    for i in range(0, len(id_arr)):
        id_arr[i] += offset + 1
        heads_arr[i] += offset + 1
    return [id_arr, heads_arr]


def add_positional_links(id_arr, heads_arr):
    """
    This function adds positional links (previous word, next word) between tokens in a dependency graph.

    :param id_arr: A NumPy array containing the ids of tokens.
    :param heads_arr: A NumPy array containing the head token indices.

    :return: A unique NumPy array with the ids and heads stacked vertically.
    """
    num_nodes = id_arr.shape[0]
    for i in range(0, num_nodes):
        if i + 1 != num_nodes:
            id_arr = np.append(id_arr, id_arr[i])
            heads_arr = np.append(heads_arr, id_arr[i + 1])
            id_arr = np.append(id_arr, id_arr[i + 1])
            heads_arr = np.append(heads_arr, id_arr[i])

    to_return = np.vstack((id_arr, heads_arr))
    return to_return

def forest_from_token_dep_dict(tok_dep_dict, sentence_self_loop=False, add_sep=None):
    """
    This function combines token and dependency information from a dictionary into a forest structure.

    :param tok_dep_dict: A dictionary containing two key-value pairs for each sentence in the input string:
        * "{idx} tokens": A NumPy array containing the tokens of the sentence at index idx.
        * "{idx} dependency": A NumPy array representing the dependency links in the sentence at index idx.
    :param sentence_self_loop: A boolean indicating whether the graph has the out-edge from the sentence-representing note entering in the node itself.
    :param add_sep: A string to be inserted between sentences (if provided).

    :return: (dict) A dictionary containing the combined forest information:
            * "tokens": A list containing all tokens from all sentences.
            * "dependency": A NumPy array containing the concatenated dependency links for all sentences.
            * "first_nodes": A list containing the starting node index for each sentence.
    """
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
        if add_sep is not None and i != num_sentences-1:
              all_tokens.append(add_sep)
    return {"tokens": all_tokens, "dependency": np.array(all_dep), "first_nodes":first_nodes}

def get_forest_from_sentence(to_tokenize, add_sep=None, sentence_self_loop=False, positional_links=True):
    """
    This function takes a sentence as input and returns a forest representation containing tokens, dependencies,
    and starting node indices for each sentence (if multiple sentences are present).

    :param to_tokenize: The sentence to be tokenized and converted into a forest structure.
    :param add_sep: A string to be inserted between sentences (if provided).
    :param sentence_self_loop: A boolean indicating whether the graph has the out-edge from the sentence-representing note entering in the node itself.
    :param positional_links: A boolean flag indicating whether to add edges between tokens based
                                on their relative position in the sentence (e.g., previous word, next word).

    :return: (dict) A dictionary containing the combined forest information:
            * "tokens": A list containing all tokens from all sentences.
            * "dependency": A NumPy array containing the concatenated dependency links for all sentences.
            * "first_nodes": A list containing the starting node index for each sentence.
    """
    return forest_from_token_dep_dict(get_tokens_and_dependencies(to_tokenize,
                                                                  sentence_self_loop,
                                                                  positional_links),
                                      sentence_self_loop, add_sep)

class GloveUtils:
    """
    This class provides utilities for loading and working with GloVe embeddings.

    Attributes:
        :type glove_path: str
            The path to the GloVe text file containing word embeddings.
        :type vocabulary: dict
            A dictionary of current vocabulary, mapping words to their corresponding GloVe embedding vectors.
        :type embeddings_dict: dict
            A dictionary mapping words to their corresponding GloVe embedding vectors.
        :type pca: sklearn.decomposition.PCA
            A PCA object used for dimensionality reduction (if enabled).
        :type max_proj: float
            The maximum projected value after applying PCA.
        :type min_proj: float
            The minimum projected value after applying PCA.
        :type vocab_path: str (optional)
            The path to the file for storing or loading the vocabulary dictionary.
            Defaults to "./glove_vocab.pkl".
        :type embed_dimension: int (optional)
            The dimension of the GloVe word embeddings (assumed to be 100 by default).
    """
    def __init__(self, glove_path, vocab_path="./glove_vocab.pkl"):
        """
        Initializes a GloveUtils object.

        :param glove_path: The path to the file containing GloVe word embeddings.
        :param vocab_path: The path to the file for storing or loading the vocabulary dictionary.
        """
        self.glove_path = glove_path
        self.vocabulary = {}
        self.embeddings_dict = {}
        self.pca = PCA(n_components=3)
        self.max_proj = 0
        self.min_proj = 0
        self.vocab_path = vocab_path
        self.embed_dimension = 100

        # Load GloVe embeddings from file
        if os.path.exists(glove_path):
            with open(glove_path, 'r', encoding="utf-8") as f:
              for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        else:
            print("GloVe path missing!")

        # Load vocabulary from file (if it exists)
        if os.path.exists(self.vocab_path):
            self.load_vocab(self.vocab_path)
        else:
            print("Saving directory missing!")

    def embed_to_GloVe(self, tokens_to_embed, pca_flag=False):
        """
        Embeds a list of tokens using GloVe word embeddings.

        :param tokens_to_embed: A list of tokens to be embedded.
        :param pca_flag: A flag indicating whether to apply PCA dimensionality reduction

        :return: A NumPy array containing the word embeddings for the input tokens.
        """
      # Initialize an empty list to store the embeddings
      embeddings = []

      # Iterate through the tokens and retrieve the embeddings
      for token in tokens_to_embed:
          if token in self.embeddings_dict:
              embeddings.append(self.embeddings_dict[token])
              self.vocabulary.update({token:self.embeddings_dict[token]})
          else:
            # If the token is not found in the vocabulary, you can assign a random embedding or any other handling strategy
            random_embed = np.random.uniform(-0.25, 0.25, self.embed_dimension)
            random_embed = random_embed.astype("float32")
            embeddings.append(random_embed)
            self.vocabulary.update({token:random_embed})

      # Convert the list of embeddings to a NumPy array
      if pca_flag:
          self.__fit_pca__()
      embeddings = np.array(embeddings)
      return embeddings

    def __fit_pca__(self):
        """
        Fits the PCA object to the current vocabulary if not already fitted.
        """
        to_fit = [self.vocabulary[word] for word in list(self.vocabulary.keys())]
        fitted_vocab = self.pca.fit_transform(to_fit)
        self.max_proj = np.amax(fitted_vocab)
        self.min_proj = np.amin(fitted_vocab)

    def project_embedding(self, embed_to_project):
        """
        Projects a single word embedding using the fitted PCA object.

        :param embed_to_project: A NumPy array representing the word embedding to be projected.

        :return: A NumPy array containing the projected word embedding (normalized using min and max values after PCA).
        """
        projected_embed = self.pca.transform(np.array([embed_to_project]))
        to_return = (projected_embed - self.min_proj) / (self.max_proj - self.min_proj)
        return to_return

    def project_embeddings(self, embeddings):
        """
        Projects a set of word embeddings using the fitted PCA object.

        :param embeddings: A NumPy array containing the word embeddings to be projected.

        :return: A NumPy array containing the projected word embeddings (normalized using min and max values after PCA).
        """
        self.__fit_pca__()
        projected_embed = self.pca.transform(embeddings)
        to_return = (projected_embed - self.min_proj) / (self.max_proj - self.min_proj)
        return to_return

    def project_tokens(self, tokens_to_project):
        """
        Projects a list of tokens using GloVe embeddings and PCA.

        :param tokens_to_project: A list of tokens to be embedded and projected.

        :return: A NumPy array containing the projected word embeddings.
        """
        n_tok = len(tokens_to_project)
        embeddings = self.embed_to_GloVe(tokens_to_project)
        to_return = np.zeros((n_tok, 3))
        for i in range(0, n_tok):
            to_return[i, :] = self.project_embedding(embeddings[i, :])
        return to_return

    def serialize_vocab(self, path):
        """
        Serializes the vocabulary dictionary to a file using pickle.

        :param path: (str) The path to the file where the vocabulary dictionary will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vocabulary, f)

    def load_vocab(self, path, pca_flag=False):
        """
        Loads the vocabulary dictionary from a file using pickle (if the file exists).
        Optionally fits the PCA object if the flag is True.

        :param path: The path to the file containing the serialized vocabulary dictionary.
        :param pca_flag: A flag indicating whether to fit the PCA object based on the loaded vocabulary
        """
        with open(path, 'rb') as f:
            self.vocabulary = pickle.load(f)
        if pca_flag:
           self.__fit_pca__()

# Initialize GloveUtils object using glove.twitter.27B.100d
glove_path = './glove.twitter.27B.100d.txt'
g_utils = GloveUtils(glove_path)

# Initialize Bert tokenizer and model
bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# class containing the graph data that must be fed into the GCNs or GAT networks
class Dataset_from_sentences(Dataset):
    """
    This class represents a dataset of graph data for GCN or GAT networks, constructed from sentences and labels.

    Attributes:
        :type name: str
            The name of the dataset.
        :type drive_dir: str
            The path to a Google Drive directory containing the raw dataset file.
        :type root: str
            The root directory where the processed dataset will be saved.
        :type raw_url: str
            The URL of the raw dataset file.
        :type data_list: list
            A list of PyTorch Geometric `Data` objects representing the processed graph data points.
        :type sentences_list: list (optional)
            A list of sentences to be used for processing (if not loaded from a file).
        :type y_values: list (optional)
            A list of labels corresponding to the sentences.
        :type embedding: str
            The type of word embedding to be used ("glove" or "bert").
    """
    def __init__(self, name, path_where_save, drive_dir, sentences_list=None, y_values=None, embedding="glove", transform=None):
        """
        Initializes a Dataset_from_sentences object.

        :param name: The name of the dataset.
        :param path_where_save: The root directory where the processed dataset will be saved.
        :param drive_dir: The path to a Google Drive directory containing the raw dataset file.
        :param sentences_list: A list of sentences to be used for processing.
        :param y_values: A list of labels corresponding to the sentences.
        :param embedding: The type of word embedding to be used ("glove" or "bert").
        :param transform: A transformation to be applied to the data.
        """
        self.name = name
        self.drive_dir = drive_dir
        self.root = path_where_save
        self.raw_url = str(self.drive_dir + self.name + ".pt")
        self.data_list = []
        print(self.raw_paths)

        if sentences_list != None:
        self.sentences_list = sentences_list

        if y_values != None:
        self.y_values = y_values

        self.embedding = embedding
        if os.path.exists(self.raw_paths[0]):
        # Load data list from saved file
        self.data_list = torch.load(self.raw_paths[0])
        self.data_list.x = self.data_list.x.type(torch.FloatTensor)
        self._indices = range(self.len())
        self.transform = None
        else:
        # Download data if not found locally
        print(f"Missing local data in {path_where_save}, downloading...")
        super().__init__(path_where_save, transform)

    @property
    def processed_file_names(self):
        """
        Returns the path to the processed dataset file.
        """
        return self.raw_paths[0]

    @property
    def raw_paths(self):
        """
        Returns a list containing the path to the processed dataset file.
        """
        to_return = self.root + "/" + self.name + ".pt"
        return [to_return]

    def download(self):
        """
        Downloads the dataset from Google Drive if possible.
        """        
        if os.path.exists(self.raw_url):
            shutil.copy(self.raw_url, self.raw_paths[0])
        else:
            pass

    def len(self):
        """
        Returns the number of data in the dataset.
        """
        return len(self.data_list)

    def get(self, idx):
        """
        Returns the data point at index "idx"
        """
        return self.data_list[idx]

    def __build_graph_Data_with_GloVe__(self, sentence, y_val, return_dep_dict=False, pca_flag=False):
        """
        Constructs a graph data point using GloVe word embeddings for the given sentence.

        :param sentence: The sentence to be processed.
        :param y_val: The label of the sentence.
        :param return_dep_dict: A flag indicating whether to return a dictionary containing dependency information.
        :param pca_flag: A flag indicating whether to perform PCA on the GloVe embeddings.

        :return: A PyTorch Geometric `Data` object or a tuple containing the `Data` object and the dependency dictionary (if return_dep_dict is True).
        """
        # Extract dependency information
        tok_dep_dict = get_forest_from_sentence(sentence)
        intra_sentence_data_list = []

        # Process sentence if it has tokens
        if len(tok_dep_dict.keys()) > 0:
            # Get GloVe and sentence embeddings
            glove_embeddings = torch.tensor(g_utils.embed_to_GloVe(tok_dep_dict["tokens"], pca_flag))
            glove_sentence_avg = torch.mean(glove_embeddings, dim=0)

            # the sentence is modelled as the directed dependency graph where the nodes
            # have the bert embeddings as features, since the verb of the main
            # sentence points to 0, node 0 has the pooler output as its features,
            # as they represent the meaning of the whole sentence

            # Combine sentence embedding with individual word embeddings
            node_features = torch.vstack((glove_sentence_avg, glove_embeddings))
            reshaped = glove_sentence_avg.reshape(1, list(glove_sentence_avg.shape)[0])
            for first_node in tok_dep_dict["first_nodes"]:
                if first_node != 0:
                    node_features = torch.cat((node_features[0:first_node, :],
                              reshaped, node_features[first_node:, :]), 0)

            # Get edge indices from dependency information
            edge_idxs = torch.tensor(tok_dep_dict[f"dependency"], dtype = torch.int64)
            data = Data(x = node_features, edge_index = edge_idxs, y = y_val)
            intra_sentence_data_list.append(data)
            batch = Batch.from_data_list(intra_sentence_data_list)

            if return_dep_dict:
                return batch, tok_dep_dict
        return batch

    def __build_graph_Data_with_BERT__(self, sentence, y_val, bert_model, bert_tok, return_dep_dict = False):
        """
        Constructs a graph data point using BERT word embeddings for the given sentence.

        :param sentence: The sentence to be processed.
        :param y_val: The label of the sentence.
        :param bert_model: The BERT model to be used for embedding.
        :param bert_tok: The tokenizer for the BERT model.
        :param return_dep_dict: A flag indicating whether to return a dictionary containing dependency information.
        
        :return: A PyTorch Geometric `Data` object or a tuple containing the `Data` object and the dependency dictionary (if return_dep_dict is True).
        """
        # Extract dependency information
        tok_dep_dict = get_forest_from_sentence(sentence, add_sep="[SEP]")
        intra_sentence_data_list = []

        # Process sentence if it has tokens
        if len(tok_dep_dict.keys()) > 0:
            # Get BERT embeddings
            node_features = self._get_bert_embeddings(tok_dep_dict["tokens"], bert_model, bert_tok)

            # the sentence is modelled as the directed dependency graph where the nodes
            # have the bert embeddings as features, since the verb of the main
            # sentence points to 0, node 0 has the pooler output as its features,
            # as they represent the meaning of the whole sentence

            # Check for potential errors in the embedding size
            if node_features.shape[0] != len(tok_dep_dict["tokens"]) + 1:
              print("Error with sentence:")
              print(sentence)

            # Get edge indices from dependency information
            edge_idxs = torch.tensor(tok_dep_dict[f"dependency"], dtype = torch.int64)
            data = Data(x = node_features, edge_index = edge_idxs, y = y_val)
            intra_sentence_data_list.append(data)
            batch = Batch.from_data_list(intra_sentence_data_list)

            if return_dep_dict:
                return batch, tok_dep_dict
        return batch

    def _get_bert_embeddings(self, tokens, model, tokenizer):
        """
        Creates the tensor from with the bert embeddings of the sentences in tokens.
        The output contains as first embedding the pooler output.
        The encodings of sub-words are combined with a mean to obtain an approximate
        encoding for the full token.

        :param tokens: A list of tokens to be embedded. Must contain "sep" tokens in case of multiple sentences.
        :param model: The Bert model to use for embeddings
        :param tokenizer: The tokenizer to use for the model

        :return: A tensor of dimension [len(tokens) + 1, embedding dimension]
        """
        encoded = tokenizer([" ".join(tokens)])
        with torch.inference_mode():
            output = model(**encoded.convert_to_tensors("pt"))
            embeddings = output.last_hidden_state.squeeze()
            pooler = output.pooler_output.squeeze()
        word_embeddings = pooler
        for idx, word in enumerate(tokens):
            start, end = encoded.word_to_tokens(idx)
            word_embeddings = torch.vstack((word_embeddings, embeddings[start:end].mean(dim=0)))
        return word_embeddings

    def to(self, device):
        """
        Moves the object to the specified device (CPU or GPU).
        """
        self.data_list.to(device)
        return self

    def process(self):
        """
        Processes the sentence list and builds graph data objects.
        """
        num_invalid = 0
        for idx, elem in tqdm(enumerate(self.sentences_list)):
            if self.embedding == "glove":
                to_save = self.__build_graph_Data_with_GloVe__(elem, self.y_values[idx])
            elif self.embedding == "bert":
                to_save = self.__build_graph_Data_with_BERT__(elem, self.y_values[idx], bert_model, bert_tok)
            else:
                raise NotImplementedError(f"Embedding type not supported: {self.embedding}")
            if to_save is not None:
                self.data_list.append(to_save)
            else:
                num_invalid += 1
        torch.save(Batch.from_data_list(self.data_list), self.raw_paths[0])
        torch.save(Batch.from_data_list(self.data_list), self.raw_url)

        if self.embedding == "glove":
            g_utils.serialize_vocab(g_utils.vocab_path)

    def normalize_and_save(self):
        """
        Normalizes the data points list and saves it.
        """
        self.data_list.x = normalize(self.data_list.x, p = 1, dim = 1)
        torch.save(self.data_list, self.raw_paths[0])
        torch.save(self.data_list, self.raw_url)


# function needed to clear a directory, it is necessary to call this function
# if some modifications to the dataset must be performed
def delete_processed_files(directory):
    """
    Deletes all files and subdirectories within a given directory.

    :param directory: The path to the directory to be cleared.
    """
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

def visualize_graph(G, td_dict=[]):
    """
    Visualizes a graph using NetworkX and matplotlib.

    :param G: The Pytorch Geometric graph object to be visualized.
    :param td_dict: A dictionary containing information about the graph's nodes.
            - `td_dict["first_nodes"]` (list): A list of indices representing the "sentence" nodes.
            - `td_dict["tokens"]` (list): A list of tokens corresponding to the nodes (excluding the sentence node).
            If not provided, the function assumes no labels for the nodes.
    """
    to_draw = to_networkx(G, to_undirected = False)
    embeds = G["x"]
    embeds = embeds.cpu()
    n_nodes = embeds.shape[0]

    # Calculate node colors based on projected embeddings
    colors = np.power(g_utils.project_embeddings(embeds), 2)
    color = "red"

    plt.figure(figsize=(7,7)) # Set figure size
    plt.xticks([]) # Hide x-axis ticks
    plt.yticks([]) # Hide y-axis ticks
    
    if td_dict == []:
        # No node labels provided
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

    # Display the graph
    plt.show()

def visualize_hidden_graph(x_features, links, labels=[], project_flag=True, k_custom=None):
    """
    Build and visualize a directed graph using NetworkX and matplotlib.

    :param x_features: A tensor containing node features.
    :param links: A tensor with two rows representing source and target nodes for edges.
    :param labels: A list of labels for each node.
    :param project_flag: A flag indicating whether to project node embeddings before coloring the nodes.
    :param k_custom: A custom parameter for the spring layout algorithm in NetworkX.
    """
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
        nx.draw_networkx(to_draw, pos=nx.spring_layout(to_draw, seed=42, k=k_custom), with_labels=False,
                         node_color=colors, cmap="Set2")
    else:
        labels = dict(zip(np.arange(1, n_nodes), labels))
        labels.update({0:"Sentence"})
        nx.draw_networkx(to_draw, pos=nx.spring_layout(to_draw, seed=42, k=k_custom), with_labels=True,
                         labels=labels,
                         node_color=colors, cmap="Set2")
    plt.show()

def print_example(example_sentence):
    """
    Function printing an example sentence with the respective NER labels obtained with stanza.

    :param example_sentence: string containing the sample sentence to print
    """
    nlp_example_sentence = nlp(example_sentence)
    to_print = []
    for sentence in nlp_example_sentence.sentences:
        cur_str = ""
        for word in sentence.words:
            cur_str = f"id: {word.id}\thead id: {word.head}\tdependency relation: {word.deprel}\tword: {word.text}"
            to_print.append(cur_str)
    for elem in to_print:
        print(elem)


print("Utils have been correctly loaded")
