import os
from collections import defaultdict
from sklearn.model_selection import train_test_split


def remove_noisy_label(label, prev_label, next_label):
    if label in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']:
        return label
    else:
        if (prev_label is None) or (next_label is None):
            return 'O'
        if label.endswith('MISC'):
            return 'O'
        if prev_label.startswith('B'):
            if prev_label == 'B-PER':
                return 'I-PER'
            elif prev_label == 'B-LOC':
                return 'I-LOC'
            else:
                return 'I-ORG'
        elif next_label.startswith('I'):
            if prev_label == 'O':
                if next_label == 'I-PER':
                    return 'B-PER'
                elif next_label == 'I-LOC':
                    return 'B-LOC'
                else:
                    return 'B-ORG'
            elif prev_label.startswith('I'):
                return prev_label
        else:
            return 'O'


def load_data(data_path, data_format='1', test_size=0.15):
    """
    Load data from file and split training set and test set
    :param data_path: path to data folder
    :param test_size: the ratio of test set to total dataset,
        0 < test_size < 1, default = 0.2
    :param data_format: form of a output data point:
        '1': ('word', 'ner_label')
        '2.1': ('word', 'pos-tagging label')
        '2.2': ('word', 'chucking label', 'ner label')
        '3': ('word', 'pos-tagging label', 'chucking label', 'ner label')
        default = '1'
    :return: training_set, test_set
    """
    # Check input
    if data_format not in {'1', '2.1', '2.2', '3'}:
        raise Exception("{} not is a data_format. The value of data_format should in ('1', '2.1', '2.2', '3')"
                        .format(data_format))
    if not os.path.exists(data_path):
        raise Exception("{} does not exist" .format(data_path))
    if test_size <= 0 or test_size >= 1:
        raise Exception("Test_size should be between 0 and 1. The value of test_size is: {}" .format(test_size))

    # load data
    data = []
    for file_name in os.listdir(data_path):
        file_path = data_path + '/' + file_name
        with open(file_path, encoding='utf-8') as f:
            sentence = []
            all_data = f.readlines()
            for i in range(len(all_data)):
                line = all_data[i]
                label = line.split()
                if len(label) > 4:
                    # prev_label = all_data[i - 1].split()[3] if len(sentence) != 0 else None
                    # next_label = all_data[i + 1].split()[3] if len(all_data[i + 1].split()) > 4 else None
                    # label[3] = remove_noisy_label(label[3], prev_label, next_label)
                    if label[3] not in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']:
                        continue
                    # all_data[i] = label[0] + '	' + label[1] + '	' + label[2] + '	' + label[3] + '	' + label[4]
                    if data_format == '1':
                        sentence.append((label[0], label[3]))
                    elif data_format == '2.1':
                        sentence.append((label[0], label[1], label[3]))
                    elif data_format == '2.2':
                        sentence.append((label[0], label[2], label[3]))
                    else:
                        sentence.append((label[0], label[1], label[2], label[3]))
                else:
                    data.append(sentence)
                    sentence = []
        f.close()

    # split training set and test set
    training_set, test_set = train_test_split(data, test_size=test_size, random_state=123, shuffle=True)
    return training_set, test_set


def create_transition_matrix_bigram(dataset, data_path):
    transition_matrix = create_dict()
    transition_matrix['start'] = defaultdict(int)
    for sentence in dataset:
        for i in range(len(sentence)):
            ner = sentence[i][1]
            if i == 0:
                transition_matrix['start'][ner] += 1
            elif i == len(sentence) - 1:
                prev_ner = sentence[i - 1][1]
                transition_matrix[prev_ner][ner] += 1
                transition_matrix[ner]['stop'] += 1
            else:
                prev_ner = sentence[i - 1][1]
                transition_matrix[prev_ner][ner] += 1
    freq_matrix = defaultdict()
    with open(data_path, 'w') as f:
        for prev_ner in transition_matrix.keys():
            curr_ner = transition_matrix[prev_ner]
            total_label = sum(curr_ner.values())
            freq_bigram = {}
            print(str(prev_ner) + ': ' + str(total_label))
            for ner in curr_ner.keys():
                # print(curr_ner[ner], total_label)
                freq_bigram[ner] = curr_ner[ner] / total_label
                f.write((prev_ner + '<fff>' + ner + '<fff>' + str(freq_bigram[ner])))
                f.write('\n')
                # print((prev_ner + '<fff>' + ner + '<fff>' + str(freq_bigram[ner]) + '\n'))
            freq_matrix[prev_ner] = freq_bigram
    f.close()


def create_dict():
    dct = defaultdict()
    dct['B-PER'] = defaultdict(int)
    dct['I-PER'] = defaultdict(int)
    dct['B-ORG'] = defaultdict(int)
    dct['I-ORG'] = defaultdict(int)
    dct['B-LOC'] = defaultdict(int)
    dct['I-LOC'] = defaultdict(int)
    dct['O'] = defaultdict(int)
    return dct


def create_transition_matrix_trigram(dataset, data_path):
    transition_matrix = defaultdict()
    transition_matrix['start'] = create_dict()
    transition_matrix['start']['start'] = defaultdict(int)
    transition_matrix['B-PER'] = create_dict()
    transition_matrix['I-PER'] = create_dict()
    transition_matrix['B-ORG'] = create_dict()
    transition_matrix['I-ORG'] = create_dict()
    transition_matrix['B-LOC'] = create_dict()
    transition_matrix['I-LOC'] = create_dict()
    transition_matrix['O'] = create_dict()
    for sentence in dataset:
        for i in range(len(sentence)):
            ner = sentence[i][1]
            if i == 0:
                transition_matrix['start']['start'][ner] += 1
            elif i == 1:
                prev1_ner = sentence[i - 1][1]
                transition_matrix['start'][prev1_ner][ner] += 1
            elif i == len(sentence) - 1:
                prev2_ner = sentence[i - 2][1]
                prev1_ner = sentence[i - 1][1]
                transition_matrix[prev2_ner][prev1_ner][ner] += 1
                transition_matrix[prev1_ner][ner]['stop'] += 1
            else:
                prev2_ner = sentence[i - 2][1]
                prev1_ner = sentence[i - 1][1]
                transition_matrix[prev2_ner][prev1_ner][ner] += 1
    freq_matrix = defaultdict()
    with open(data_path, 'w') as f:
        for prev2_ner in transition_matrix.keys():
            trigram = transition_matrix[prev2_ner]
            freq_trigram = {}
            for prev1_ner in trigram.keys():
                # print(curr_ner[ner], total_label)
                bigram = trigram[prev1_ner]
                freq_bigram = {}
                total_label = sum(bigram.values())
                for curr_ner in bigram.keys():
                    freq_bigram[curr_ner] = bigram[curr_ner] / total_label
                    f.write((prev2_ner + '<fff>' + prev1_ner + '<fff>' + curr_ner + '<fff>' + str(freq_bigram[curr_ner])))
                    f.write('\n')
                freq_trigram[prev1_ner] = freq_bigram
            freq_matrix[prev2_ner] = freq_trigram
    f.close()


def create_emission_probability(dataset, data_path):
    emission = create_dict()
    vocab_size = defaultdict()
    for sentence in dataset:
        for i in range(len(sentence)):
            word = sentence[i][0]
            ner = sentence[i][1]
            emission[ner][word] += 1
    emission_probability = defaultdict()
    with open(data_path, 'w', encoding='utf-8') as f:
        for ner in emission.keys():
            words = emission[ner]
            total_label = sum(words.values())
            vocab_size[ner] = len(words)
            # print(vocab_size)
            emission_bigram = {}
            for word in words.keys():
                # print(ner[ner], total_label)
                emission_bigram[word] = (words[word] + 1) / (total_label + vocab_size[ner])
                f.write((word + '<fff>' + ner + '<fff>' + str(emission_bigram[word])))
                f.write('\n')
            emission_probability[ner] = emission_bigram
    f.close()


def training():
    path = 'G:/Project/python/NER/data/NER2016-training_data'
    train, test = load_data(path)

    emission_path = 'G:/Project/python/NER/data/bigram_data'
    create_emission_probability(train, emission_path + '/emission_probability_train.txt')
    create_transition_matrix_bigram(train, emission_path + '/transition_matrix_train.txt')

    emission_path = 'G:/Project/python/NER/data/trigram_data'
    create_emission_probability(train, emission_path + '/emission_probability_train.txt')
    create_transition_matrix_trigram(train, emission_path + '/transition_matrix_train.txt')
    return test


# if __name__ == '__main__':
#     path = 'G:/Project/python/NER/data/NER2016-training_data'
#     train, test = load_data(path)
#     data = train + test
#
#     emission_path = 'G:/Project/python/NER/data/count'
#     create_transition_matrix_bigram(data, emission_path + '/transition_matrix_train.txt')
    # create_transition_matrix_trigram(train, emission_path + '/transition_matrix_train.txt')
