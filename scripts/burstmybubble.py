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
from bokeh.palettes import Spectral5
from bokeh.transform import factor_cmap
from bokeh.io import export_png
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


def explore_csv(csv, outprefix):
    """
    Explore a csv file with a dataframe wiith the following columns:
    'Title', 'Content', 'Objectivism', 'Outlet', 'Bias'
    :param outprefix: prefix of output plot
    :param csv: csv filename
    :return:
    """
    df = pd.read_csv(csv, header=None, names=[
        'Title', 'Content', 'Objectivism', 'Outlet', 'Bias'])
    gr = df.groupby('Bias')['Title'].size()
    source = ColumnDataSource({'Bias': gr.index.tolist(), 'count': gr.values})
    try:
        bias = source.data['Bias'].tolist()
    except AttributeError:
        bias = source.data['Bias']
    if all([str(i).isdigit() for i in bias]):
        # Categories are numeric and need to be categorical. For the training
        # set of the all news 0: neutral, 1: left, 2: right
        d = {0: 'neutral', 1: 'left', 2: 'right'}
        bias = [d[x] for x in bias]
        source.data['Bias'] = bias
    p = figure(x_range=bias)
    color_map = factor_cmap(field_name='Bias', palette=Spectral5, factors=bias)
    p.vbar(x='Bias', top='count', source=source, width=0.70, color=color_map)
    p.title.text = "Publications per bias in %s" % csv
    p.xaxis.axis_label = 'Bias'
    p.yaxis.axis_label = 'Number of publications'
    # remove toolbar (no point in still image)
    p.toolbar.logo = None
    p.toolbar_location = None
    export_png(p, filename="%s.png" % outprefix)


class process_n_train(object):
    def __init__(self, data_path, ngram, input_type='cornell', sample=False):
        """
        Class to process speech usinh NLP (ngram), remove invariants,
        and train a Lda model
        :param data_path: Path to where the textfiles are
        """
        self.sample = sample
        self.data_path = data_path
        if input_type == 'cornell':
            # assume multiple files formatted as convote
            self.files = glob(os.path.join(data_path, '*.txt'))
        else:
            assert (data_path.endswith('db') or data_path.endswith('csv'))
            self.files = data_path
        self.input_type = input_type
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.bias = None
        self.info = {}
        self.sentences = self.files
        self.bow = ngram
        self.lda_transf = None
        self.lda_fit = None
        self.pca_array = None
        self.pca()
        self.lda()

    @property
    def sentences(self):
        return self.__sentences

    @sentences.setter
    def sentences(self, files):
        if self.input_type == 'cornell':
            # Cornell-type input assumes a directory with multiple txt files
            self.bias, self.__sentences, self.info = self.read_files_cornell()
            # Transform the list of dictionary into a dictionary of lists
            temp = defaultdict(list)
            for element in self.info:
                for k, v in element.items():
                    temp[k].append(v)
            self.info = temp
        elif self.input_type == 'news_audit':
            # News_audit assumes a csv file (see readme in github)
            self.bias, self.__sentences, self.info = self.read_csv_table()
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
        if self.sample:
            df = pd.DataFrame(text_counts.toarray()).sample(frac=0.01, axis=1)
        else:
            df = pd.DataFrame(text_counts.toarray())
        self.__bow = df

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

    def read_csv_table(self):
        """
        Helper to read news_audit-formatted csv files
        :return: tuple with political bias list, list of documents, and info
        """
        df = pd.read_csv(self.files, header=None, names=[
            'Title', 'Content', 'Objectivism', 'Outlet', 'Bias'])
        df = df.dropna(subset=['Content'])
        if all([str(i).isdigit() for i in df.Bias]):
            # Categories are numeric and need to be categorical. For the
            # training set of the all news 0: neutral, 1: left, 2: right
            d = {0: 'neutral', 1: 'left', 2: 'right'}
            df.Bias = [d[x] for x in df.Bias]
        if self.sample:
            df = df.sample(frac=0.1)
        info = {'Objectivism': df.Objectivism, 'Outlet': df.Outlet,
                'Title': df.Title}
        return df.Bias, df.Content, info

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
        colors = dict(zip(set(self.bias), colors))
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
        d['colors'] = [colors[b] for b in self.bias]
        source = ColumnDataSource(data=d)
        TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)")] + \
                   [(k, '@%s' % k) for k in d.keys() if
                    k not in ['x', 'y', 'colors']]
        p = figure(plot_width=600, plot_height=300, tooltips=TOOLTIPS)
        p.scatter('x', 'y', source=source, radius=1.5, fill_color='colors',
                  fill_alpha=0.6, line_color=None)
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        output_file('%s_%s_plot.bokeh.html' % (self.input_type, plot), mode='inline')
        save(p)


def main(data_path, ngram, input_type, sample):
    train = process_n_train(data_path, ngram, input_type=input_type,
                            sample=sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_path', help='path to data',
                        default='../data/raw/convote_v1.1/data_stage_two/'
                                'development_set')
    parser.add_argument('--ngram', type=int, nargs=2,
                        help='ngrams to use. This takes to values for the '
                             'range of ngrams to be explored')
    parser.add_argument('--input_type', default='cornell',
                        help='Kind of input. For now only cornell and '
                             'news_audit (see readme on github)')
    parser.add_argument('--sample', action='store_true', default=False,
                        help='Sample BOW to 10% for testing purposes')
    args = parser.parse_args()
    main(args.data_path, tuple(args.ngram), args.input_type, sample=args.sample
         )
