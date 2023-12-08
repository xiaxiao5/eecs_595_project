import pandas as pd
import os
import json
from allennlp.predictors.predictor import Predictor
import numpy as np
import ast
import os
import time 

def prepare_data(config):
    if config['dataset'] == "Ego4D":
        # assum v2 is downloaded. --dataset full_scale annotations clips

        # read original narration annotation file
        # read og narrations
        with open(pd.read_json(os.path.join(config['dataset_root_folder'], 'Ego4D/raw/v2/annotations/narration.json')), 'r') as f:
            og_narration = json.load(f)
        max_vid_num = 3
        all_narration_df = pd.DataFrame(columns=['vid_id', 'pass_name', 'timestamp_frame', 'narration_text'])
        for vid_no, (vid_id, vid_val) in enumerate(og_narration.items()):
            if vid_no > max_vid_num:
                break
            for pass_name, pass_val in vid_val.items():
                if pass_name == 'status':
                    continue
                for narration_info in pass_val['narrations']:
                    narration_text = narration_info['narration_text']

                    timestamp_frame = narration_info['timestamp_frame']
                    all_narration_df.loc[len(all_narration_df)] = {'vid_id': vid_id, 'pass_name': pass_name, 'timestamp_frame': timestamp_frame, 'narration_text': narration_text}
        # e.g.
        #         vid_id	                            pass_name	timestamp_frame	narration_text
        # 0	77cc4654-4eec-44c6-af05-dbdf71f9a401	narration_pass_1	0	        #C C interacts with a woman X
        # 1	77cc4654-4eec-44c6-af05-dbdf71f9a401	narration_pass_1	136	        #C C walks into the kitchen
        

        # make srl.csv: apply SRL to each narration

        # make groups.csv: group narrations based on SRL and the original taxonomy
        # read taxonomy
        taxonomy_verb_path = os.path.join(config['dataset_root_folder'], 'Ego4D/raw/v2/annotations/narration_verb_taxonomy.csv')
        taxonomy_verb_df = pd.read_csv(taxonomy_verb_path)
        taxonomy_verb_df['group'] = taxonomy_verb_df['group'].apply(ast.literal_eval)
        taxonomy_noun_path = os.path.join(config['dataset_root_folder'], 'Ego4D/raw/v2/annotations/narration_noun_taxonomy.csv')
        taxonomy_noun_df = pd.read_csv(taxonomy_noun_path)
        taxonomy_noun_df['group'] = taxonomy_noun_df['group'].apply(ast.literal_eval)
        # grouping
        grouping_results = all_narration_df.apply(grouping, axis=1)
        grouping_results_df = pd.DataFrame(grouping_results.tolist(), columns=['verb_group', 'noun_group'])
        # some nouns are not grouped since the matching is not perfect (string based not semantic based or find the dic containing all plurality). Could be better
        # grouping_results_df.isna().sum() / len(grouping_results_df)
        # concat the results with the original narration df
        grouped_narration_df = pd.concat([all_narration_df, grouping_results_df], axis=1)
        # drop rows with nan (not grouped narrations)
        clean_narration_df = grouped_narration_df.dropna(axis=0)

        # make annotation.csv: rows are groups, columns are lists of groups that are different from the row gorup in with different chunks (S, A, O1, O2...)


        # make difference.csv: an adjacent matrix of the groups. Each field indicates the number of semantic chunks that are different between the two groups

def srl(config):
    # load config
    if config['dataset'] == 'ego4d':
        narration_path = os.path.join(config['dataset_root_proj'], 'annotation', 'egoclip.csv') # download from egovlp repo
    device = config['device']
    srl_inference = config['srl_inference']

    # load narration
    all_narration_df = pd.read_csv(narration_path, sep='\t', on_bad_lines='warn')
    narration_column_name = 'clip_text'

    # pre-process narration text
    # del start #C/#O. e.g., "#C C opens the washing machine door" -> "C opens the washing machine door"
    all_narration_df[narration_column_name] = all_narration_df[narration_column_name].str.split(' ', n=1).str[1]
    # skip those narrations containing '#unsure'. --- has been cleaned by egoclip
    # all_narration_df = all_narration_df[-all_narration_df[narration_column_name].str.contains('#unsure')]
    # replace C with 'The recorder' -- no difference
    # all_narration_df[narration_column_name] = all_narration_df[narration_column_name].str.replace('C ', 'The recorder ')
    # drop nan (one narration is nan in egoclip.csv)
    all_narration_df = all_narration_df[~all_narration_df[narration_column_name].isna()].reset_index()

    # apply SRL
    all_narration_list = all_narration_df[narration_column_name].tolist()
    batch_size = 10000
    if srl_inference:
        slrpredictor = SRLPredictor(device=device)
        print(f'Applying SRL to {len(all_narration_df)} narrations in batches')
        
        assert batch_size <= len(all_narration_list)
        for sample_i in range(0, len(all_narration_list), batch_size):
            start_time = time.time()
            srl_results_batch = slrpredictor.predict(all_narration_list[sample_i : sample_i+batch_size])
            print(f'DONE SRL for batch {sample_i/batch_size+1}/{len(all_narration_list)//batch_size+1} in {time.time()-start_time:.2f}s')
            # save tmp results
            srl_batch_save_path = os.path.join(config['dataset_root_proj'], 'annotation', f'srl_b{sample_i:07d}.json')
            with open(srl_batch_save_path, 'w') as f:
                json.dump(srl_results_batch, f, indent=4)
            print(f'batch{sample_i} file saved in {srl_batch_save_path}')

    # start_time = time.time()
    # srl_results = slrpredictor.predict(all_narration_df[narration_column_name].tolist())
    # print(f'DONE SRL in {time.time()-start_time:.2f}s')
    # # save tmp results
    # with open('/tmp/srl.json', 'w') as f:
    #     json.dump(srl_results, f, indent=4)
    # print(f'tmp file saved in /tmp/srl.json')

    # read all srl results lists
    srl_results = []
    for sample_i in range(0, len(all_narration_list), batch_size):
        srl_batch_save_path = os.path.join(config['dataset_root_proj'], 'annotation', f'srl_b{sample_i:07d}.json')        
        with open(srl_batch_save_path, 'r') as f:
            srl_batch_save_batch = json.load(f)
            srl_results += srl_batch_save_batch

    # srl_results is a list of dict. 
    # e.g. {'verbs': [{'verb': 'picks', 'description': '[ARG0: C] [V: picks] [ARG1: a bag of clothes] [ARG2: from the floor]', 'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2', 'I-ARG2']}], 'words': ['C', 'picks', 'a', 'bag', 'of', 'clothes', 'from', 'the', 'floor']}

    # phrase into dataframe
    srl_df = {}
    for sentence_no, srl_result in enumerate(srl_results):
        # match role and word
        try:
            srl_matching = list(zip(srl_result['verbs'][0]['tags'], srl_result['words'])) # e.g., {'B-ARG0': '#', 'I-ARG0': 'C', 'B-V': 'interacts', 'B-ARG1': 'with', 'I-ARG1': 'X'}
        except:
            # if no verb is found
            srl_matching = [('NO_VERB_DET', 'True')]
        # make dataframe columns
        for srl_full_tag, srl_word in srl_matching:
            srl_tag = srl_full_tag.split('-')[-1]
            if srl_tag not in srl_df:
                srl_df[srl_tag] = {}
            if sentence_no not in srl_df[srl_tag]:
                srl_df[srl_tag][sentence_no] = []
            srl_df[srl_tag][sentence_no].append(srl_word)
    srl_df = pd.DataFrame(srl_df)
    # srl_df e.g., 
	# ARG0	V	ARG1	ARG2	DIR	LOC	ARG3	MNR	EXT	ARG4	O	GOL	PRP	ADV	COM	ARG5	PRD
    # 0	[C]	[picks]	[a, bag, of, clothes]	[from, the, floor]	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN

    # make SRL chunk of into strings
    srl_df = srl_df.apply(lambda col: col.str.join(' '), axis=0)

    # concat narration and SRL
    complete_srl_df = pd.concat([all_narration_df, srl_df], axis=1)

    # save
    output_path = os.path.join(config['dataset_root_proj'], 'annotation', 'egoclip_srl.csv')
    complete_srl_df.to_csv(output_path, index=False)



class SRLPredictor:
    def __init__(self, device) -> None:
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
                                cuda_device=0 if device == "cuda" else -1)
    def predict(self, sentences):
        instances = [{"sentence": sentence} for sentence in sentences]
        results = self.predictor.predict_batch_json(instances)
        return results        


# defin grouping function
def grouping(narr_row, taxonomy_verb_df, taxonomy_noun_df):
    # given a narration row (narration_text), return the valid noun group
    # the tagged verb group is always correct, but the tagged noun group may tag some words from other than ARG1
    tag_noun = narr_row['tag_noun'] # list
    ARG1 = narr_row['ARG1'] # string
    noun_group = []


    # noun group
    for noun_group_row_i in tag_noun:
        noun_group_row = taxonomy_noun_df.iloc[noun_group_row_i]
        noun_options = noun_group_row['group']
        for noun_option in noun_options: # noun_option is a list of nouns in the taxonomy group
            if check_words_in(ARG1, noun_option) and noun_group_row_i not in noun_group:
                noun_group.append(noun_group_row_i)
    # if noun_group == []:
    #     noun_group = tag_noun
    return noun_group


def check_words_in(main_string, substr):
    # check if substr is in main_string, and if it is a word in the main string..
    # e.g., check_words_in('a cat eats food', 'cat') -> True
    # e.g., check_words_in('a cat eats food', 'a cat') -> True
    # e.g., check_words_in('a cat eats food', 'ca') -> False
    start_idx = main_string.find(substr)
    # If substring not found, return None
    if start_idx == -1:
        return False
    
    left_char = main_string[start_idx - 1] if start_idx != 0 else '|'
    right_char = main_string[start_idx + len(substr)] if start_idx + len(substr) < len(main_string) else '|'

    if left_char in ['|', ' '] and right_char in ['|', ' ', 's', 'es']:
        return True
    else:
        return False