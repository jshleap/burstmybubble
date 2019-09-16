"""
**burstmybubble.py** Classifying news bias
**Copyright** 2019  Jose Sergio Hleap
"""
__author__ = 'Jose Sergio Hleap'
__email__ = "jshleap@gmail.com"
__version__ = '0.1b'


import os
import sqlite3
import argparse
from glob import glob

import pandas as pd
from bokeh.plotting import figure, output_file, ColumnDataSource, save
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.feature_extraction.text import CountVectorizer


class process_n_train(object):
    def __init__(self, data_path, ngram, type='cornell'):
        """
        Class to process speech usinh NLP (ngram), remove invariants,
        and train a Lda model
        :param data_path: Path to where the textfiles are
        """
        self.data_path = data_path
        if type == 'cornell':
            # assume multiple files formatted as convote
            self.files = glob(os.path.join(data_path, '*.txt'))
        else:
            assert data_path.endswith('db')
            self.files = data_path
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.type = type
        self.bias = None
        self.info = {}
        self.sentences = self.files
        self.bow = ngram
        self.lda_transf = None
        self.lda_fit = None
        self.pca_array = None
        self.lda()

    @property
    def sentences(self):
        return self.__sentences

    @sentences.setter
    def sentences(self, files):
        if self.type == 'cornell':
            self.bias, self.__sentences, self.info = self.read_files_cornell()
        else:
            self.bias, self.__sentences, self.info = self.read_db_table()

    @property
    def bow(self):
        return self.__bow

    @bow.setter
    def bow(self, ngram):
        """
        Compute BOW with given ngram range
        """
        cv = CountVectorizer(
            lowercase=True, stop_words='english',  ngram_range=ngram,
            tokenizer=self.tokenizer.tokenize
        )
        text_counts = cv.fit_transform(self.sentences)
        self.__bow = pd.DataFrame(text_counts.toarray())

    def read_files_cornell(self):
        """
        Given a list of files formatted as the convote of cornell (see readme
        for more details) extract meaningful information
        :return: tuple with political bias list, list of documents, and info
        """
        parties = []
        sentences = []
        info = []
        for filename in self.files:
            # get label based on the filename structure:
            # ###_@@@@@@_%%%%$$$_PMV.txt, where p is the party
            parts = os.path.basename(filename[:filename.rfind('.txt')]).split(
                '_')
            info.append(dict(party=parts[-1][0], vote=parts[-1][-1],
                             bill=parts[1], speaker=parts[2]))
            parties.append(parts[-1][0])
            with open(filename) as text:
                sentences.append(text.read())
        return parties, sentences, info

    def read_db_table(self):
        """
        Read table from Fake news detection, Google Summer of Code 2017
        http://www.newsaudit.net. This is a different dataset from that
        explained in the readme.
        :return: tuple with political bias list, list of documents, and info
        """
        db = sqlite3.connect(self.files)
        query = '''SELECT * FROM longform;'''
        df = pd.read_sql_query(query, db)
        info = dict(
            title=df.title.tolist(), author=df.author.tolist(),
            date=df.date.tolist(), year=df.year.tolist(),
            month=df.month.tolist(), publication=df.publication.tolist()
        )
        return df.left, df.content, info

    def pca(self, **kwargs):
        """
        Fit BOW to the fisrt two PC
        """
        print('Data has %d rows and %d columns' % (self.bow.shape[0],
                                                   self.bow.shape[1]))
        pca = PCA(n_components=2)
        self.pca_array = pca.fit_transform(self.bow)
        self.pca_bokeh_plot(plot='pca')

    def lda(self):
        """
        Fit and transform BOW dataframe creating a discriminant function
        :return: fitted instance and transformed array
        """
        print('Data has %d rows and %d columns' % (self.bow.shape[0],
                                                   self.bow.shape[1]))
        lda = Lda(n_components=None)
        self.lda_fit = lda.fit(self.bow, y=self.bias)
        self.lda_transf = self.lda_fit.transform(self.bow)
        self.pca_bokeh_plot()

    def pca_bokeh_plot(self, plot='lda'):
        """
        Interactive PCA plot using Bokeh
        """
        colors = ['crimson', 'navy', 'green']
        colors = dict(zip(colors, self.bias))
        if plot == 'lda':
            t_pca = self.lda_transf
        else:
            t_pca = self.pca_array
        if t_pca.shape[1] == 1:
            d = dict(x=range(t_pca.shape[0]), y=t_pca[:, 0])
            xlabel = 'Document Index'
            ylabel = 'Dimension 1'
        else:
            d = dict(x=t_pca[:, 0], y=t_pca[:, 1])
            xlabel = 'Dimension 1'
            ylabel = 'Dimension 2'
        # Set the information to be displayed in the interactive plot
        d.update(self.info)
        d['colors'] = colors
        source = ColumnDataSource(data=d)
        TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)")] + \
                   [(k, '@%s' % k) for k in d.keys() if
                    k not in ['x', 'y', 'colors']]
        p = figure(plot_width=600, plot_height=300, tooltips=TOOLTIPS)
        p.scatter('x', 'y', source=source, radius=1.5, fill_color='colors',
                  fill_alpha=0.6, line_color=None)
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        output_file('%s_plot.bokeh' % plot)
        save(p)


def main(data_path, ngram):
    train = process_n_train(data_path, ngram)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_path', help='path to data',
                        default='../data/raw/convote_v1.1/data_stage_two/'
                                'development_set')
    parser.add_argument('--ngram', type=tuple, nargs=2,
                        help='ngrams to use. This takes to values for the '
                             'range of ngrams to be explored'
                        )
    args = parser.parse_args()
    main(args.data_path, args.ngram)