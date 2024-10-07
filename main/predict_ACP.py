import os
import sys
import esm
import math
import torch
import argparse
import collections
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(current_dir)

def esm_embeddings(peptide_sequence_list):
  # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long, 
  #         or you have too many sequences for transformation in a single converting, 
  #         you conputer might automatically kill the job.
  # load the model
  # NOTICE: if the model was not downloaded in your local environment, it will automatically download it.
  model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
  batch_converter = alphabet.get_batch_converter()
  model.eval()  # disables dropout for deterministic results

  # load the peptide sequence list into the bach_converter
  batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
  ## batch tokens are the embedding results of the whole data set

  # Extract per-residue representations (on CPU)
  with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t33_650M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
      results = model(batch_tokens, repr_layers=[6], return_contacts=True)  
  token_representations = results["representations"][6]

  # Generate per-sequence representations via averaging
  # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
  sequence_representations = []
  for i, tokens_len in enumerate(batch_lens):
      sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
  # save dataset
  # sequence_representations is a list and each element is a tensor
  embeddings_results = collections.defaultdict(list)
  for i in range(len(sequence_representations)):
      # tensor can be transformed as numpy sequence_representations[0].numpy() or sequence_representations[0].to_list
      each_seq_rep = sequence_representations[i].tolist()
      for each_element in each_seq_rep:
          embeddings_results[i].append(each_element)
  embeddings_results = pd.DataFrame(embeddings_results).T
  return embeddings_results


def extract_feature(sequence_list):
    # load sequence for esm-2
    peptide_sequence_list = []
    for seq in sequence_list:
        format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information

    # employ ESM model for converting and save the converted data in csv format
    embeddings_results = esm_embeddings(peptide_sequence_list)

    return embeddings_results


def text_fasta_reading(file_name):
    """
    A function for reading txt and fasta files
    """
    # read txt file with sequence inside
    file_read = open(file_name, mode='r')
    file_content = []  # create a list for the fasta content temporaty storage
    for line in file_read:
        file_content.append(line.strip())  # extract all the information in the file and delete the /n in the file

    # build a list to collect all the sequence information
    sequence_name_collect = collections.defaultdict(list)
    for i in range(len(file_content)):
        if '>' in file_content[i]:  # check the symbol of the
            sequence_name_collect[file_content[i]].append(file_content[i + 1])

    # transformed into the same style as the xlsx file loaded with pd.read_excel and sequence_list = dataset['sequence']
    sequence_name_collect = pd.DataFrame(sequence_name_collect).T
    sequence_list = sequence_name_collect[0]
    return sequence_list


# collect the output
def assign_result(predicted_class):
    out_put = []
    for i in range(len(predicted_class)):
        if predicted_class[i] == 0:
            # out_put[int_features[i]].append(1)
            out_put.append('ACP')
        else:
            # out_put[int_features[i]].append(2)
            out_put.append('non-ACP')
    return out_put


def predict(feature):
    # Create a directory to save the model
    model_GRU_dir = 'ACP/save_models/GRU/Independence'
    model_CNN_dir = 'ACP/save_models/CNN/Independence'
    model_CapsuleGAN_dir = 'ACP/save_models/CapsuleGAN/Independence'

    all_predictions = []
    for i in range(10):
        # Loading model
        if i == 0:
            [sample_num, input_dim] = np.shape(feature)
            X = np.reshape(feature, (-1,1,input_dim))
            model_path = os.path.join(model_GRU_dir, f'ESM_{i}.h5')
            clf = load_model(model_path)
        elif i in (1, 2, 3, 4, 5, 6, 7):
            [sample_num, input_dim] = np.shape(feature)
            X = np.reshape(feature, (-1,1,input_dim))
            model_path = os.path.join(model_CNN_dir, f'ESM_{i}.h5')
            clf = load_model(model_path)
        else:
            X = feature
            model_path = os.path.join(model_CapsuleGAN_dir, f'ESM_{i}.h5')
            clf = load_model(model_path)

        y_score = clf.predict(X)
        all_predictions.append(y_score)


    # 转换为 numpy 数组
    all_predictions = np.array(all_predictions)

    # 平均投票
    average_predictions = np.mean(all_predictions, axis=0)
    final_predictions = np.argmax(average_predictions, axis=1)

    return final_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process FASTA files and predict ACPs.')
    parser.add_argument('fasta_file', type=str, help='Path to the input FASTA file')

    args = parser.parse_args()

    # Reading FASTA files
    sequence_list = text_fasta_reading(args.fasta_file)

    # Extract features
    embeddings = extract_feature(sequence_list)

    # Making predictions
    predicted_class = predict(embeddings)

    result = assign_result(predicted_class)

    # Printing Results
    print(result)
    