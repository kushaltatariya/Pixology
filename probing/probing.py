
from __future__ import absolute_import, division, unicode_literals

import io

import numpy as np
from tqdm import tqdm

from splitclassifier import SplitClassifier


class Probing:
    def __init__(self, task, params, batcher, layer):
        self.seed = "1111"
        self.params = params
        self.batcher = batcher
        self.task = task
        self.layer = layer
        self.task_data = {'train': {'X': [], 'y': []},
                          'dev': {'X': [], 'y': []},
                          'test': {'X': [], 'y': []}}

        self.loadFile(f"{self.params['task_dir']}/{self.task}.txt")
        # results = self.run(self.params, self.batcher, self.task)
        # return results

    def loadFile(self, fpath):
        self.tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split('\t')
                # self.task_data[self.tok2split[line[0]]]['X'].append(line[-1].split())
                self.task_data[self.tok2split[line[0]]]['X'].append(line[-1])
                self.task_data[self.tok2split[line[0]]]['y'].append(line[1])

        labels = sorted(np.unique(self.task_data['train']['y']))
        self.tok2label = dict(zip(labels, range(len(labels))))
        self.nclasses = len(self.tok2label)

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split]['y']):
                try:
                    self.task_data[split]['y'][i] = self.tok2label[y]
                except:
                    print(y)
                    print(self.task_data[split]['X'][i])
                    quit()

    def run(self, params, batcher, task):
        task_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params["batch_size"]

        print(f'Computing embeddings for train/dev/test for {self.task}')

        for key in self.task_data:
            indexes = list(range(len(self.task_data[key]['y'])))
            layer_embs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [],
                          8: [], 9: [], 10: [], 11: [], 12: []}

            sorted_data = sorted(zip(self.task_data[key]['X'],
                                     self.task_data[key]['y'], indexes),
                                 key=lambda z: (len(z[0]), z[1], z[2]))

            self.task_data[key]['X'], self.task_data[key]['y'], self.task_data[key]['idx'] = map(list,
                                                                                                 zip(*sorted_data))

            task_embed[key]['X'] = {}

            for i in tqdm(range(0, len(self.task_data[key]['y']), bsize)):
                batch = self.task_data[key]['X'][i:i + bsize]
                embs = batcher(params, batch, task)
                for k, v in embs.items():
                    layer_embs[k].append(embs[k])
            for layer in range(1, 13):
                task_embed[key]['X'][layer] = np.vstack(layer_embs[layer])
                task_embed[key]['y'] = np.array(self.task_data[key]['y'])
                task_embed[key]['idx'] = np.array(indexes)
        print("Computed embeddings!")

        assert task_embed['train']['X'][1].shape[0] == task_embed['train']['y'].shape[0] == \
               task_embed['train']['idx'].shape[0]

        params_classifier = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                             'tenacity': 5, 'epoch_size': 4}
        config_classifier = {'nclasses': self.nclasses, 'seed': 1223,
                             'usepytorch': True,
                             'classifier': params_classifier}

        results = {}
        if self.layer == 'all':
            for layer in tqdm(range(1, 13)):
                print(f"Training classifier on embeddings from layer {layer}...")
                clf = SplitClassifier(X={'train': task_embed['train']['X'][layer],
                                         'valid': task_embed['dev']['X'][layer],
                                         'test': task_embed['test']['X'][layer]},
                                      y={'train': task_embed['train']['y'],
                                         'valid': task_embed['dev']['y'],
                                         'test': task_embed['test']['y']},
                                      config=config_classifier)

                devacc, testacc, predictions = clf.run()
                results[layer] = (devacc, testacc, predictions)
                print(('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (results[layer][0], results[layer][1], self.task.upper())))

        else:
            print(f"Training classifier on embeddings from layer {self.layer}...")
            clf = SplitClassifier(X={'train': task_embed['train']['X'][self.layer],
                                     'valid': task_embed['dev']['X'][self.layer],
                                     'test': task_embed['test']['X'][self.layer]},
                                  y={'train': task_embed['train']['y'],
                                     'valid': task_embed['dev']['y'],
                                     'test': task_embed['test']['y']},
                                  config=config_classifier)

            devacc, testacc, predictions = clf.run()
            results[self.layer] = (devacc, testacc, predictions)
            print(('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (results[self.layer][0], results[self.layer][1], self.task.upper())))

        return results