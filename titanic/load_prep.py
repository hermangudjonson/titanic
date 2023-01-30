"""
Routines to load and prep Titanic data
"""
import pandas as pd
import numpy as np
import scipy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

# raw data loading
def raw_train():
    """load raw training data
    
    Returns
    -------
    train_df: `pd.DataFrame`
        891 x 10, mixed numeric and categorical
    target_ds: `pd.Series`
        891 Survived (binary 0/1)
    """
    train_df = pd.read_csv('data/raw/train.csv', index_col='PassengerId')
    target_ds = train_df['Survived']
    train_df = train_df.drop(columns='Survived') # remove target from training features
    return train_df, target_ds

def raw_test():
    """load raw test data
    
    Returns
    -------
    test_df: `pd.DataFrame`
        418 x 10
    """
    test_df = pd.read_csv('data/raw/test.csv', index_col='PassengerId')
    return test_df
    
def example_submission():
    """load example submission, gender prediction
    """
    gender_ds = pd.read_csv('data/raw/gender_submission.csv', index_col='PassengerId').squeeze()
    return gender_ds


# transform data columns
class TicketTransformer(BaseEstimator, TransformerMixin):
    """
    learned state from fit is ticket group survival and size
    """
    def __init__(self):
        self.line_id = 0
        self.t_count = None
        self.t_survived = None

    def _make_line_unique(self, df):
        # make LINE unique, start from current line_id
        K = sum(df.Ticket == 'LINE')
        df.loc[df.Ticket=='LINE','Ticket'] = [f'LINE{i}' for i in range(self.line_id, self.line_id + K)]
        self.line_id += K
        
    def fit(self, df, y=None):
        # watch out for LINE before aggregating
        self._make_line_unique(df)

        # ticket group survival count and size count
        self.t_count = y.groupby(df['Ticket']).agg(t_survived='sum', t_total='count')
        # passengers and survival part of stored ticket groups
        self.t_survived = y
        return self

    def transform(self, df):
        self._make_line_unique(df)

        # ticket grouping
        # create a feature indicating the survival rate of other members of a ticket group (in training)
        # can think of these as ticket group neighbors
        # try encoding as an adjustment to the baseline survival rate
        # to account for ticket group sample size, using bayesian bernoulli

        # add survival info if available
        df = df.merge(self.t_survived, how='left', left_index=True, right_index=True)

        # long form, create column entry for each passenger
        t_count_long = df[['Survived','Ticket']].merge(self.t_count, how='left', left_on='Ticket', right_index=True)
        # potentially missing ticket groups 
        t_count_long.fillna({'t_survived':0, 't_total':0}, inplace=True)
        # adjust so passenger's own counts don't contribute
        t_count_long['nt_survived'] = t_count_long.t_survived - t_count_long.Survived.fillna(0)
        t_count_long['nt_total'] = t_count_long.t_total - ~t_count_long.Survived.isna()

        # bayesian bernoulli
        smean = self.t_survived.mean()
        ssize = 5 # wide gaussian around mean
        t_beta = scipy.stats.beta(
            smean * ssize + t_count_long.nt_survived, 
            (1 - smean) * ssize + t_count_long.nt_total - t_count_long.nt_survived
        )
        # survival rate adjustment relative to mean
        t_count_long['Ticket_Survival'] = t_beta.mean() - smean
        # match original index
        t_count_long = t_count_long.loc[df.index,:]
        df['Ticket_Survival'] = t_count_long.Ticket_Survival

        df = df.drop(columns='Survived')

        return df


def _transform_ticket(df):
    """old routine, this is 'correct' but could leak test data
    """
    # make LINE unique
    df.loc[df.Ticket=='LINE','Ticket'] = [f'LINE{i}' for i in range(sum(df.Ticket=='LINE'))]
    
    # ticket grouping
    # create a feature indicating the survival rate of other members of a ticket group (in training)
    # can think of these as ticket group neighbors
    # try encoding as an adjustment to the baseline survival rate
    # to account for ticket group sample size, using bayesian bernoulli
    t_count = df.groupby('Ticket').Survived.agg(t_survived='sum', t_total='count')
    # aggregated count per ticket group
    # long form, create column entry for each passenger
    t_count_long = df[['Survived','Ticket']].merge(t_count, left_on='Ticket', right_index=True)
    # adjust so passenger's own counts don't contribute
    t_count_long['nt_survived'] = t_count_long.t_survived - t_count_long.Survived.fillna(0)
    t_count_long['nt_total'] = t_count_long.t_total - ~t_count_long.Survived.isna()
    
    # bayesian bernoulli
    smean = df.Survived.mean()
    ssize = 5 # wide gaussian around mean
    t_beta = scipy.stats.beta(
        smean * ssize + t_count_long.nt_survived, 
        (1 - smean) * ssize + t_count_long.nt_total - t_count_long.nt_survived
    )
    # survival rate adjustment relative to mean
    t_count_long['Ticket_Survival'] = t_beta.mean() - smean
    # match original index
    t_count_long = t_count_long.loc[df.index,:]
    df['Ticket_Survival'] = t_count_long.Ticket_Survival
    
def transform_name(df):
    # regex to extract last name and title
    name_title = df.Name.str.extract(r'(\w+),\s([\w\s]+)\.')
    name_title.columns = ['Last_Name','Title']
    
    # remap some extraneous title categories
    tdict = {
        'Capt': 'Professional',
        'Col': 'Professional',
        'Don': 'Mr',
        'Dona': 'Mrs',
        'Dr': 'Professional',
        'Jonkheer': 'Professional',
        'Lady': 'Mrs',
        'Major': 'Professional',
        'Master': 'Master',
        'Miss': 'Miss',
        'Mlle': 'Miss',
        'Mme': 'Miss',
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Ms': 'Miss',
        'Rev': 'Professional',
        'Sir': 'Mr',
        'the Countess': 'Mrs',
    }
    df['Title'] = name_title.Title
    df = df.replace({'Title': tdict})
    df['Title'] = pd.Categorical(df.Title)
    return df
    
def transform_sibsp_parch(df):
    df['Family_Size'] = df.SibSp + df.Parch + 1
    return df

def transform_fare(df):
    # map 0 values to NA
    df = df.replace({'Fare':0.0}, np.nan)
    # log transform 
    df['Fare'] = np.log2(df.Fare + 1)
    return df

def transform_cabin(df):
    df['Deck'] = df.Cabin.str.extract(r'([A-Z])\w*', expand=False)
    df['Deck'] = df.Deck.fillna('Z')
    df['Deck'] = pd.Categorical(df.Deck, sorted(df.Deck.unique().tolist()))
    return df

def transform_embarked(df):
    df['Embarked'] = df.Embarked.fillna('S')
    df['Embarked'] = pd.Categorical(df.Embarked)
    return df

def transform_sex(df):
    df['Sex'] = pd.Categorical(df.Sex)
    return df

def select_columns(df):
    # post preprocessing select relevant columns
    keep_cols = ['Pclass','Deck','Fare','Sex','Age','Title','Family_Size','Ticket_Survival']
    df = df.loc[:, keep_cols]
    return df

def preprocess_pipeline():
    """construct sklearn pipeline for Titanic preprocessing steps

    performs:
     - basic feature correction
     - feature rescaling
     - new feature creation
     - categorical datatype definition
     - old feature removal

    Returns
    -------
    pipe: `Pipeline`
        Titanic to transform raw input to preprocessed input
        ready for imputation, categorical encoding and classification
    """
    scale_numeric = make_column_transformer(
        (StandardScaler(), ['Fare','Age']), 
        remainder='passthrough', 
        verbose_feature_names_out=False
    )
    scale_numeric.set_output(transform='pandas')

    pipe = make_pipeline(
        # stateless
        FunctionTransformer(transform_sex),
        FunctionTransformer(transform_embarked),
        FunctionTransformer(transform_cabin),
        FunctionTransformer(transform_fare),
        FunctionTransformer(transform_sibsp_parch),
        FunctionTransformer(transform_name),
        # custom state
        TicketTransformer(),
        # scale age and fare
        scale_numeric,
        # select columns
        FunctionTransformer(select_columns),
        
    )
    return pipe

def transform_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
):
    """Titanic feature transformation pipeline

    performs:
     - basic feature correction
     - feature rescaling
     - new feature creation
     - categorical datatype definition
     - old feature removal

    Parameters
    ----------
    train_df: `pd.DataFrame`
        raw training data (N x 11) including `Survived` column
    test_df: `pd.DataFrame`
        raw test data (N x 10)

    Returns
    -------
    new_train_df: `pd.DataFrame`
        new training data (N x 9) including `Survived` column with transformed features
    new_test_df: `pd.DataFrame'
        new test data (N x 8) with transformed features
    """
    # combine train and test for feature work
    traintest_df = pd.concat([train_df, test_df])

    transform_embarked(traintest_df)
    transform_cabin(traintest_df)
    transform_fare(traintest_df)
    transform_sibsp_parch(traintest_df)
    transform_name(traintest_df)
    _transform_ticket(traintest_df)

    keep_cols = ['Survived','Pclass','Deck','Fare','Sex','Age','Title','Family_Size','Ticket_Survival']
    ntrain_df = traintest_df.loc[train_df.index, keep_cols]
    ntest_df = traintest_df.loc[test_df.index, keep_cols[1:]]
    return ntrain_df, ntest_df