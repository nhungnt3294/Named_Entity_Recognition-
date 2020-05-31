from collections import defaultdict
from train_HMM import training, create_dict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class HMM_NER:
    def __init__(self):
        self.trigram = defaultdict()
        self.trigram['start'] = create_dict()
        self.trigram['start']['start'] = defaultdict(int)
        self.trigram['B-PER'] = create_dict()
        self.trigram['I-PER'] = create_dict()
        self.trigram['B-ORG'] = create_dict()
        self.trigram['I-ORG'] = create_dict()
        self.trigram['B-LOC'] = create_dict()
        self.trigram['I-LOC'] = create_dict()
        self.trigram['O'] = create_dict()

        self.bigram = create_dict()
        self.bigram['start'] = defaultdict(int)

        self.emission = create_dict()
        self.states = {'B-PER': 0,
                       'I-PER': 1,
                       'B-LOC': 2,
                       'I-LOC': 3,
                       'B-ORG': 4,
                       'I-ORG': 5,
                       'O': 6}
        self.eval_states1 = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
        self.eval_states = [0, 1, 2, 3, 4, 5]

    def load_test_data(self, test_set):
        x = []
        y_word = []
        for sentence in test_set:
            if len(sentence) <= 0:
                continue
            xi = []
            yi = []
            for i in range(len(sentence)):
                xi.append(sentence[i][0])
                yi.append(self.states[sentence[i][1]])
            x.append(xi)
            y_word.extend(yi)
        return x, y_word

    def load_trigram(self, trigram_path):
        for u in self.states.keys():
            for v in self.states.keys():
                for w in self.states.keys():
                    self.trigram[u][v][w] = 0
                    self.trigram['start']['start'][w] = 0
                    self.trigram['start'][v][w] = 0
        with open(trigram_path) as f:
            for line in f.readlines():
                line = line.strip()
                y1, y2, y3, probability = line.split('<fff>')
                self.trigram[y1][y2][y3] = probability

    def load_bigram(self, bigram_path):
        self.bigram['start']['start'] = 0
        for u in self.states.keys():
            for v in self.states.keys():
                self.bigram[u][v] = 0
            self.bigram['start'][u] = 0
        with open(bigram_path) as f:
            for line in f.readlines():
                line = line.strip()
                y1, y2, probability = line.split('<fff>')
                self.bigram[y1][y2] = probability

    def load_emission(self, emission_path):
        with open(emission_path, encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                observation, state, probability = line.split('<fff>')
                # self.emission[state] = defaultdict()
                # self.emission[state][observation] = 0
                self.emission[state][observation] = probability

    def bigram_transition_probability(self, y1, y2):
        return float(self.bigram[y1][y2])

    def trigram_transition_probability(self, y1, y2, y3):
        return float(self.trigram[y1][y2][y3])

    def state_observation_likelihood(self, observation, state):
        return float(self.emission[state][observation])

    def rare_word_observation(self, state):
        # print(1 / len(self.emission[state].keys()))
        return 1 / len(self.emission[state].keys())

    def bigram_decode(self, sentence):
        viterbi = [defaultdict() for _ in range(len(sentence))]
        back_point = [defaultdict() for _ in range(len(sentence) + 1)]
        tag_seq = [0 for _ in range(len(sentence) + 1)]
        word = sentence[0]
        for state in self.states.keys():
            # print(self.bigram_transition_probability('start', state))
            # print(self.state_observation_likelihood(sentence[0], state))
            # print('----------------')
            state_vocab = [key for key in self.emission[state].keys()]
            if word in state_vocab:
                viterbi[0][state] = self.bigram_transition_probability('start', state) \
                                    * self.state_observation_likelihood(word, state)
            else:
                viterbi[0][state] = self.bigram_transition_probability('start', state) \
                                    * self.rare_word_observation(state)
            # print(state + ': ' + str(self.bigram['start'][state]))
            back_point[0][state] = 0
            # tag_seq[0] = 0
        for i in range(1, len(sentence)):
            word = sentence[i]
            for v in self.states.keys():
                max_score = 0
                tag = None
                for u in self.states.keys():
                    # print(self.emission[v].keys())
                    if word in self.emission[v].keys():
                        # print(1)
                        score = viterbi[i - 1][u] \
                                * self.bigram_transition_probability(u, v) \
                                * self.state_observation_likelihood(word, v)
                    else:
                        score = viterbi[i - 1][u] \
                                * self.bigram_transition_probability(u, v) \
                                * self.rare_word_observation(v)
                    if score > max_score:
                        max_score = score
                        tag = self.states[u]
                viterbi[i][v] = max_score
                back_point[i][v] = tag
                # tag_seq[i] = tag

        max_score = 0
        tag = None
        for u in self.states.keys():
            score = viterbi[len(sentence) - 1][u] \
                    * self.bigram_transition_probability(u, 'stop')
            if score > max_score:
                max_score = score
                tag = self.states[u]
        # if tag is None:
        #     print(max_score)
        best_prob = max_score
        tag_seq[len(sentence)] = tag
        back_point[len(sentence)]['stop'] = tag

        for k in range(len(sentence) - 2, 0, -1):
            print(tag_seq)
            # print(k)
            ner = (list(self.states.keys()))[list(self.states.values()).index(tag_seq[k + 1])]
            tag_seq[k] = back_point[k][ner]
        # print(tag_seq)

        return best_prob, tag_seq[1:]

    def trigram_decode(self, sentence):
        viterbi = [defaultdict() for _ in range(len(sentence))]
        back_point = [defaultdict() for _ in range(len(sentence) + 1)]
        tag_seq = [0 for _ in range(len(sentence) + 1)]
        word = sentence[0]
        for state in self.states.keys():
            state_vocab = [key for key in self.emission[state].keys()]
            if word in state_vocab:
                viterbi[0][state] = self.trigram_transition_probability('start', 'start', state) \
                                    * self.state_observation_likelihood(word, state)
            else:
                viterbi[0][state] = self.trigram_transition_probability('start', 'start', state) \
                                    * self.rare_word_observation(state)
            # print(state + ': ' + str(self.bigram['start'][state]))
            back_point[0][state] = [-1, -1]
            # tag_seq[0] = 0
        cnt = 0
        for i in range(1, len(sentence)):
            word = sentence[i]
            for w in self.states.keys():
                max_score = 0
                tag = None
                for v in self.states.keys():
                    for u in self.states.keys():
                        # print(self.emission[v].keys())
                        # print(word)
                        if word in self.emission[v].keys():
                            # print(1)
                            score = viterbi[i - 1][v] \
                                    * self.trigram_transition_probability(u, v, w) \
                                    * self.state_observation_likelihood(word, w)
                        else:
                            score = viterbi[i - 1][v] \
                                    * self.trigram_transition_probability(u, v, w) \
                                    * self.rare_word_observation(w)
                        if score > max_score:
                            max_score = score
                            tag = [self.states[u], self.states[v]]
                # if tag is None:
                #     print(cnt)
                #     cnt += 1
                viterbi[i][w] = max_score
                back_point[i][w] = tag
                # tag_seq[i] = tag

        max_score = 0
        tag = None
        for v in self.states.keys():
            for u in self.states.keys():
                score = viterbi[len(sentence) - 1][v] \
                        * self.trigram_transition_probability(u, v, 'stop')
                if score > max_score:
                    max_score = score
                    tag = [self.states[u], self.states[v]]
        best_prob = max_score
        # print(type(tag[1]))
        tag_seq[len(sentence)] = tag[1]
        # tag_seq[len(sentence) - 1] = tag[0]
        back_point[len(sentence)]['stop'] = tag

        for k in range(len(sentence) - 1, -1, -1):
            ner = (list(self.states.keys()))[list(self.states.values()).index(tag_seq[k + 1])]
            # print(ner)
            # print(type(back_point[k][ner]))
            tag_seq[k] = back_point[k][ner][1]
        # print(tag_seq)

        return best_prob, tag_seq[1:]

    def evaluate(self, sentences, labels, method='bigram'):
        assert method in ['bigram', 'trigram']
        y_predict = []
        for sentence in sentences:
            if method == 'bigram':
                prob, state_seq = self.bigram_decode(sentence)
                y_predict.extend(state_seq)
            if method == 'trigram':
                prob, state_seq = self.trigram_decode(sentence)
                y_predict.extend(state_seq)
        acc = accuracy_score(labels, y_predict)
        print('accuracy: ', acc)
        prec, rec, f1, support = precision_recall_fscore_support(labels, y_predict, labels=self.eval_states)
        all_prec = precision_score(labels, y_predict, labels=self.eval_states, average='micro')
        all_rec = recall_score(labels, y_predict, labels=self.eval_states, average='micro')
        all_f1 = f1_score(labels, y_predict, labels=self.eval_states, average='micro')
        print('labels:    {}'.format(self.eval_states1))
        print('precision: {}'.format([str(round(p * 100, 2)) + '%' for p in prec]))
        print('recall:    {}'.format([str(round(r * 100, 2)) + '%' for r in rec]))
        print('f1-score:  {}'.format([str(round(f * 100, 2)) + '%' for f in f1]))
        print('support:   {}'.format(support))
        print('average precision: ', all_prec)
        print('average recall: ', all_rec)
        print('average f1-score: ', all_f1)

    def predict(self, sentence, method='bigram'):
        assert method in ['bigram', 'trigram']
        y_predict = []
        if method == 'bigram':
            prob, state_seq = self.bigram_decode(sentence)
            y_predict.extend(state_seq)
        if method == 'trigram':
            prob, state_seq = self.trigram_decode(sentence)
            y_predict.extend(state_seq)
        return y_predict


if __name__ == '__main__':
    hmm = HMM_NER()
    test = training()
    x_test, y_test_word = hmm.load_test_data(test)
    hmm.load_trigram('G:/Project/python/NER/data/trigram_data/transition_matrix_train.txt')
    hmm.load_emission('G:/Project/python/NER/data/bigram_data/emission_probability_train.txt')
    # hmm.evaluate(x_test, y_test_word, method='trigram')
    hmm.load_trigram('G:/Project/python/NER/data/trigram_data/transition_matrix_train.txt')
    # hmm.evaluate(x_test, y_test_word, method='bigram')
    # sent = ['Bộ', 'Nông_nghiệp', 'và', 'Phát_triển', 'nông_thôn', 'công_bố', 'hết', 'dịch', '.']
    sent = ['Lê', 'Phúc', 'sống', 'ở', 'Hà', 'Nội', 'và', 'làm', 'ở', 'công_ty', 'Hoàn', 'Cầu']
    print(hmm.predict(sent, method='trigram'))
