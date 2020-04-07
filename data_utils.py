import codecs
import os
from sklearn.model_selection import train_test_split
from NER.config import Config

config = Config()

def remove_xml_tags(filename):
    '''
    Remove xml tag in file in data folder(raw data)
    Args:
      filename: The name of the data file in dataVLSP folder
    Return:
      File of the same name has removed xml tags in data folder
    Example:
      <editor>Vietlex team, 8-2016</editor>
      -DOCSTART-
      <s>
      Đó	P	B-NP	O	O
      là	V	B-VP	O	O
      con	Nc	B-NP	O	O
    :converted into:
      Đó	P	B-NP	O	O
      là	V	B-VP	O	O
      con	Nc	B-NP	O	O

      saved in dataVLSP folder(processed data)
    '''
    f1 = open(config.rawdata_path + filename, 'r')
    f2 = open(config.data_path + filename, 'w')
    for line in f1:
        line.strip()
        if (('<title>' in line) or line.startswith('<e') or line.startswith('-D') or line.startswith('<s>')):
            pass
        elif (line.startswith('</')):
            f2.write(line.replace(line, '\n'))
        else:
            f2.write(line)
    f1.close()
    f2.close()


def clean_data(path):
    '''
    Remove xml tags of all files in the VLSP folder
    Processed data saved in dataVLSP
    '''
    list_files = os.listdir(path)
    for file in list_files:
        remove_xml_tags(file)


def prepare_data(path, scale, index_attri):
    ''' Create training data and testing data
        Format of data: CoNLL

        Args:
          path: path of data folder
          scale: test size
          index_attri: Represents the number of attributes and the associated attribute type
            index_attri == 1 : The number of attributes = 1 - only ner label. ex: [('Huế', 'B_LOC'), ('là', 'O'), ('thành_phố', 'O'), ('đẹp', 'O')]
            index_attri == 2.1 : The number of attributes = 2(pos-tagging label, ner label). ex: [('Đó', 'P', 'O'), ('là', 'V',  'O'), ('con', 'Nc', 'O'), ('đường', 'N', , 'O')]
            index_attri = 2.2 : The number of attributes = 2(chunking label, ner label). ex: [('Đó', 'B-NP', 'O'), ('là', 'B-VP', 'O'), ('con', 'B-NP', 'O'), ('đường', 'B-NP', 'O')]
            index_attri = 3 : The number of attributes = 3(pos-tagging label,chunking, ner label). ex: [('Đó', 'P', 'B-NP', 'O'), ('là', 'V', 'B-VP', 'O'), ('con', 'Nc', 'B-NP', 'O'), ('đường', 'N', 'B-NP', 'O')]
            if index_attri not in {1,2.1,2.2,3} index_attri = 2.1
        Return:
          train_sents, test_sents

        Example of format data:
        [[('Đó', 'P', 'B-NP', 'O'), ('là', 'V', 'B-VP', 'O'), ('con', 'Nc', 'B-NP', 'O'), ('đường', 'N', 'B-NP', 'O')],
        [('Đó', 'P', 'B-NP', 'O'), ('là', 'V', 'B-VP', 'O'), ('con', 'Nc', 'B-NP', 'O'), ('đường', 'N', 'B-NP', 'O')],
        ...
        ]

    '''

    # check index_attri
    if index_attri not in {1, 2.1, 2, 2, 3}:
        index_attri = 2.1
    # split data by file
    list_files = os.listdir(path)
    train_files, test_files = train_test_split(list_files, test_size=scale)

    train_sent_data = []
    test_sent_data = []

    ''' Convert data format to CoNll '''
    # training data
    for file in train_files:
        with codecs.open(path + file, 'r', 'utf8') as f:
            sentence = []
            for line in f:
                line = line.split()
                if len(line) > 3:
                    if index_attri == 1:
                        sentence.append((line[0], line[3]))
                    elif index_attri == 2.2:
                        sentence.append((line[0], line[2], line[3]))
                    elif index_attri == 3:
                        sentence.append((line[0], line[1], line[2], line[3]))
                    else:
                        sentence.append((line[0], line[1], line[3]))
                else:
                    train_sent_data.append(sentence)
                    sentence = []
        f.close()
        # testing data
        for file in test_files:
            with codecs.open(path + file, 'r', 'utf8') as f:
                sentence = []
                for line in f:
                    line = line.split()
                    if len(line) > 3:
                        if index_attri == 1:
                            sentence.append((line[0], line[3]))
                        elif index_attri == 2.2:
                            sentence.append((line[0], line[2], line[3]))
                        elif index_attri == 3:
                            sentence.append((line[0], line[1], line[2], line[3]))
                        else:
                            sentence.append((line[0], line[1], line[3]))
                    else:
                        test_sent_data.append(sentence)
                        sentence = []
            f.close()

    return train_sent_data, test_sent_data

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# X_train = [sent2features(s) for s in train_sents]
# y_train = [sent2labels(s) for s in train_sents]

# X_test = [sent2features(s) for s in test_sents]
# y_test = [sent2labels(s) for s in test_sents]

'''
Example:
>>> X_train[0][1]
{'+1:postag': 'A',
 '+1:postag[:2]': 'A',
 '+1:word.istitle()': False,
 '+1:word.isupper()': False,
 '+1:word.lower()': 'xa',
 '-1:postag': 'N',
 '-1:postag[:2]': 'N',
 '-1:word.istitle()': False,
 '-1:word.isupper()': False,
 '-1:word.lower()': 'nước_mắt',
 'bias': 1.0,
 'postag': 'N',
 'postag[:2]': 'N',
 'word.isdigit()': False,
 'word.istitle()': False,
 'word.isupper()': False,
 'word.lower()': 'chồng',
 'word[-3:]': 'ồng'}
'''