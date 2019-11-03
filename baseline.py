#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np


def train(df):
    # Input: Raw sentence dataframe with only one column with sentences tagged with POS
    # Output: 3 pandas series encoding P(W|T), P(T|<s>) and P(T(i+1)|Ti)

    # Calculate word given the tag probabilities
    word_tag_prob = (
        df
            .assign(
            raw_split=lambda x: x['raw'].str.split())  # Split sentence into a list with elements like 'word/POS'
            .drop(['raw'], axis=1)
            .explode('raw_split')  # Put each element in separate rows
            .reset_index(drop=True)
            .assign(word_tag=lambda x: x['raw_split'].str.split('/'))  # Split the elements into word-tag pairs
            .drop(['raw_split'], axis=1)
            .assign(
            word=lambda x: x['word_tag'].str[0],  # Separate words into different column
            tag=lambda x: x['word_tag'].str[1],  # Separate tags into different column
        )
            .drop(['word_tag'], axis=1)
            .groupby(by=['word', 'tag'])
            .agg({'tag': 'count'})  # Get count of all word-tag combinations
            .rename(columns={'tag': 'count'})
            .reset_index()
            .assign(
            prob=lambda x: x['count'] / x.groupby('tag')['count'].transform(sum)  # Get probability P(W|T)
        )
            .drop(['count'], axis=1)
            .set_index(['word', 'tag'])['prob']  # Form a pandas series of probabilities
    )
    return word_tag_prob


def predict_sentence(sentence_list, word_tag_prob):

    def get_word_tag_prob(word, tag):
        # Calculate P(W|T)
        try:
            return word_tag_prob[(word, tag)]
        except:
            return 1.0

    def get_tag_set(word):
        # Get total POS tags seen during training for the given word
        try:
            return word_tag_prob[word].index.tolist()
        except:
            return ['NP']

    final_tags = []
    for word in sentence_list:
        tag_set = get_tag_set(word)
        tag_prob_dict = {tag: get_word_tag_prob(word, tag) for tag in tag_set}
        final_tags.append(max(tag_prob_dict, key=tag_prob_dict.get))

    return final_tags


def predict_text(raw_df, word_tag_prob):
    # Input: Dataframe with raw sentences tagged with true POS and other probabilities from training
    # Output: Dataframe with predicted POS tags

    def disassemble(raw_sentence):
        sentence_list_with_tags = [term.split('/') for term in raw_sentence.split()]
        sentence_list_without_tags = [term[0] for term in sentence_list_with_tags]
        true_tags = [term[1] for term in sentence_list_with_tags]
        return sentence_list_without_tags, true_tags

    def assemble(sentence_list_without_tags, tags):
        sentence_list_with_tags = [sentence_list_without_tags[i] + '/' + tags[i] for i in range(len(tags))]
        return ' '.join(sentence_list_with_tags)

    correct_tags = 0
    total_tags = 0
    predicted_sentences = []

    for i in range(len(raw_df)):
        raw_sentence = raw_df.iloc[i, 0]
        sentence_list_without_tags, true_tags = disassemble(raw_sentence)
        predicted_tags = predict_sentence(sentence_list_without_tags, word_tag_prob)
        correct_tags += (np.array(predicted_tags) == np.array(true_tags)).sum()
        total_tags += len(true_tags)
        assembled_sentence = assemble(sentence_list_without_tags, predicted_tags)
        predicted_sentences.append(assembled_sentence)

    accuracy = round((float(correct_tags) * 100) / float(total_tags), 2)
    df = pd.DataFrame(predicted_sentences, columns=['raw'])
    print('Accuracy: {}%'.format(accuracy))
    return df


if __name__ == '__main__':
    pd.options.display.max_colwidth = 300

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=['raw'])
    test_df = pd.read_csv(test_file, sep='\t', header=None, names=['raw'])

    word_tag_prob = train(train_df)

    predicted_sentences = predict_text(test_df, word_tag_prob)
