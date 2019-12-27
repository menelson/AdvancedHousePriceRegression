import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import FeatureUnion

class feat_sel(BaseEstimator, TransformerMixin):
    '''
    This transformer selects either numerical or categorical features.
    In this way we can build separate pipelines for separate data types.
    '''
    def __init__(self, dtype='numeric'):
        self.dtype = dtype

    def fit( self, X, y=None ):
        return self 

    def transform(self, X, y=None):
        if self.dtype == 'numeric':
            num_cols = X.columns[X.dtypes != object].tolist()
            return X[num_cols]
        elif self.dtype == 'category':
            cat_cols = X.columns[X.dtypes == object].tolist()
            return X[cat_cols]


class df_imputer(BaseEstimator, TransformerMixin):
    '''
    Just a wrapper for the SimpleImputer that keeps the dataframe structure
    '''
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = SimpleImputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # X is supposed to be a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled
    
    
class df_scaler(BaseEstimator, TransformerMixin):
    '''
    Wrapper of StandardScaler or RobustScaler
    '''
    def __init__(self, method='standard'):
        self.scl = None
        self.scale_ = None
        self.method = method
        if self.method == 'standard':
            self.mean_ = None
        elif method == 'robust':
            self.center_ = None
        self.columns = None  # this is useful when it is the last step of a pipeline before the model

    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scl = StandardScaler()
            self.scl.fit(X)
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.scl = RobustScaler()
            self.scl.fit(X)
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        self.columns = X.columns
        return Xscaled

    def get_feature_names(self):
        return list(self.columns)  # this is going to be useful when coupled with FeatureUnion
    

class dummify(BaseEstimator, TransformerMixin):
    '''
    Wrapper for get dummies
    '''
    def __init__(self, drop_first=False, match_cols=True):
        self.drop_first = drop_first
        self.columns = []  # useful to well behave with FeatureUnion
        self.match_cols = match_cols

    def fit(self, X, y=None):
        self.columns = []  # for safety, when we refit we want new columns
        return self
    
    def match_columns(self, X):
        miss_train = list(set(X.columns) - set(self.columns))
        miss_test = list(set(self.columns) - set(X.columns))
        
        err = 0
        
        if len(miss_test) > 0:
            for col in miss_test:
                X[col] = 0  # insert a column for the missing dummy
                err += 1
        if len(miss_train) > 0:
            for col in miss_train:
                del X[col]  # delete the column of the extra dummy
                err += 1
                
        if err > 0:
            warnings.warn('The dummies in this set do not match the ones in the train set, we corrected the issue.',
                         UserWarning)
            
        return X
        
    def transform(self, X):
        X = pd.get_dummies(X, drop_first=self.drop_first)
        if (len(self.columns) > 0): 
            if self.match_cols:
                X = self.match_columns(X)
            self.columns = X.columns
        else:
            self.columns = X.columns
        return X
    
    def get_features_name(self):
        return self.columns

class tr_numeric(BaseEstimator, TransformerMixin):
    def __init__(self, SF_room=True, bedroom=True, bath=True, lot=True, service=True):
        self.columns = []  # useful to well behave with FeatureUnion
        self.SF_room = SF_room
        self.bedroom = bedroom
        self.bath = bath
        self.lot = lot
        self.service = service
     

    def fit(self, X, y=None):
        return self
    

    def remove_skew(self, X, column):
        X[column] = np.log1p(X[column])
        return X


    def SF_per_room(self, X):
        if self.SF_room:
            X['sf_per_room'] = X['GrLivArea'] / X['TotRmsAbvGrd']
        return X


    def bedroom_prop(self, X):
        if self.bedroom:
            X['bedroom_prop'] = X['BedroomAbvGr'] / X['TotRmsAbvGrd']
            del X['BedroomAbvGr'] # the new feature makes it redundant and it is not important
        return X


    def total_bath(self, X):
        if self.bath:
            X['total_bath'] = (X[[col for col in X.columns if 'FullBath' in col]].sum(axis=1) +
                             0.5 * X[[col for col in X.columns if 'HalfBath' in col]].sum(axis=1))
            del X['FullBath']  # redundant 

        del X['HalfBath']  # not useful anyway
        del X['BsmtHalfBath']
        del X['BsmtFullBath']
        return X


    def lot_prop(self, X):
        if self.lot:
            X['lot_prop'] = X['LotArea'] / X['GrLivArea']
        return X 


    def service_area(self, X):
        if self.service:
            X['service_area'] = X['TotalBsmtSF'] + X['GarageArea']
            del X['TotalBsmtSF']
            del X['GarageArea']
        return X
    

    def transform(self, X, y=None):
        for col in ['GrLivArea', '1stFlrSF', 'LotArea']:
            X = self.remove_skew(X, col)

        X = self.SF_per_room(X)
        X = self.bedroom_prop(X)
        X = self.total_bath(X)
        X = self.lot_prop(X)
        X = self.service_area(X)

        self.columns = X.columns
        return X
    

    def get_features_name(self):
        return self.columns
    
    
class drop_columns(BaseEstimator, TransformerMixin):
    '''
    Drops columns that are not useful for the model
    The decisions come from several iterations
    '''
    def __init__(self, lasso=False, ridge=False, forest=False, xgb=False, lgb=False):
        self.columns = []
        self.lasso = lasso
        self.ridge = ridge
        self.forest = forest
        self.xgb = xgb
        self.lgb = lgb
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        to_drop = [col for col in X.columns if 'NoGrg' in col]  # dropping dummies that are redundant
        to_drop += [col for col in X.columns if 'NoBsmt' in col]

        if self.lasso:
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col]
            to_drop += [col for col in X.columns if 'HouseStyle' in col] 
            to_drop += [col for col in X.columns if 'LotShape' in col] 
            to_drop += [col for col in X.columns if 'LotFrontage' in col]
            to_drop += [col for col in X.columns if 'GarageYrBlt' in col] 
            to_drop += [col for col in X.columns if 'GarageType' in col] 
            to_drop += ['OpenPorchSF', '3SsnPorch'] 
        if self.ridge: 
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col] 
            to_drop += [col for col in X.columns if 'LotFrontage' in col]
            to_drop += [col for col in X.columns if 'LotShape' in col] 
            to_drop += [col for col in X.columns if 'HouseStyle' in col] 
            to_drop += [col for col in X.columns if 'GarageYrBlt' in col]
            to_drop += [col for col in X.columns if 'GarageCars' in col] 
            to_drop += [col for col in X.columns if 'BldgType' in col] 
            to_drop += ['OpenPorchSF', '3SsnPorch']
        if self.forest: 
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col] 
            to_drop += ['OpenPorchSF', '3SsnPorch'] 
        if self.xgb:
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col]
        if self.lgb: 
            to_drop += [col for col in X.columns if 'LotFrontage' in col] 
            to_drop += [col for col in X.columns if 'HouseStyle' in col]
            to_drop += ['MisBsm'] 
            
        
        for col in to_drop:
            try:
                del X[col]
            except KeyError:
                pass
            
        self.columns = X.columns
        return X
    
    def get_feature_names(self):
        return list(self.columns)
    
    
class make_ordinal(BaseEstimator, TransformerMixin):
    '''
    Transforms ordinal features in order to have them as numeric (preserving the order)
    If unsure about converting or not a feature (maybe making dummies is better), make use of
    extra_cols and unsure_conversion
    '''
    def __init__(self, cols, extra_cols=None, include_extra='include'):
        self.cols = cols
        self.extra_cols = extra_cols
        self.mapping = {'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        self.include_extra = include_extra  # either include, dummies, or drop (any other option)
    

    def fit(self, X, y=None):
        return self
    

    def transform(self, X, y=None):
        if self.extra_cols:
            if self.include_extra == 'include':
                self.cols += self.extra_cols
            elif self.include_extra == 'dummies':
                pass
            else:
                for col in self.extra_cols:
                    del X[col]
        
        for col in self.cols:
            X.loc[:, col] = X[col].map(self.mapping).fillna(0)
        return X
    
    
class recode_cat(BaseEstimator, TransformerMixin):        
    '''
    Recodes some categorical variables according to the insights gained from the
    data exploration phase.
    '''
    def __init__(self, mean_weight=10, te_neig=True, te_mssc=True):
        self.mean_tot = 0
        self.mean_weight = mean_weight
        self.smooth_neig = {}
        self.smooth_mssc = {}
        self.te_neig = te_neig
        self.te_mssc = te_mssc
    
    
    def smooth_te(self, data, target, col):
        tmp_data = data.copy()
        tmp_data['target'] = target
        mean_tot = tmp_data['target'].mean()
        means = tmp_data.groupby(col)['target'].mean()
        counts = tmp_data.groupby(col)['target'].count()

        smooth = ((counts * means + self.mean_weight * mean_tot) / 
                       (counts + self.mean_weight))
        return mean_tot, smooth
    
    def fit(self, X, y):
        if self.te_neig:
            self.mean_tot, self.smooth_neig = self.smooth_te(data=X, target=y, col='Neighborhood')

        if self.te_mssc:
            self.mean_tot, self.smooth_mssc = self.smooth_te(X, y, 'MSSubClass')
            
        return self
    
    
    def tr_GrgType(self, data):
        data['GarageType'] = data['GarageType'].map({'Basment': 'Attchd',
                                                     'CarPort': 'Detchd',
                                                     '2Types': 'Attchd' }).fillna(data['GarageType'])
        return data
    
    
    def tr_LotShape(self, data):
        fil = (data.LotShape != 'Reg')
        data['LotShape'] = 1
        data.loc[fil, 'LotShape'] = 0
        return data
    
    
    def tr_LandCont(self, data):
        fil = (data.LandContour == 'HLS') | (data.LandContour == 'Low')
        data['LandContour'] = 0
        data.loc[fil, 'LandContour'] = 1
        return data
    
    
    def tr_LandSlope(self, data):
        fil = (data.LandSlope != 'Gtl')
        data['LandSlope'] = 0
        data.loc[fil, 'LandSlope'] = 1
        return data
    
    
    def tr_MSZoning(self, data):
        data['MSZoning'] = data['MSZoning'].map({'RH': 'RM', # medium and high density
                                                 'C (all)': 'RM', # commercial and medium density
                                                 'FV': 'RM'}).fillna(data['MSZoning'])
        return data
    
    
    def tr_Alley(self, data):
        fil = (data.Alley != 'NoAlley')
        data['Alley'] = 0
        data.loc[fil, 'Alley'] = 1
        return data
    
    
    def tr_LotConfig(self, data):
        data['LotConfig'] = data['LotConfig'].map({'FR3': 'Corner', # corners have 2 or 3 free sides
                                                   'FR2': 'Corner'}).fillna(data['LotConfig'])
        return data
    
    
    def tr_BldgType(self, data):
        data['BldgType'] = data['BldgType'].map({'Twnhs' : 'TwnhsE',
                                                 '2fmCon': 'Duplex'}).fillna(data['BldgType'])
        return data
    
    
    def tr_MasVnrType(self, data):
        data['MasVnrType'] = data['MasVnrType'].map({'BrkCmn': 'BrkFace'}).fillna(data['MasVnrType'])
        return data


    def tr_HouseStyle(self, data):
        data['HouseStyle'] = data['HouseStyle'].map({'1.5Fin': '1.5Unf',
                                                     '2.5Fin': '2Story',
                                                     '2.5Unf': '2Story',
                                                     'SLvl': 'SFoyer'}).fillna(data['HouseStyle'])
        return data


    def tr_Neighborhood(self, data):
        if self.te_neig:
            data['Neighborhood'] = data['Neighborhood'].map(self.smooth_neig).fillna(self.mean_tot)
        return data
    
    def tr_MSSubClass(self, data):
        if self.te_mssc:
            data['MSSubClass'] = data['MSSubClass'].map(self.smooth_mssc).fillna(self.mean_tot)
        return data
    
    
    def transform(self, X, y=None):
        X = self.tr_GrgType(X)
        X = self.tr_LotShape(X)
        X = self.tr_LotConfig(X)
        X = self.tr_MSZoning(X)
        X = self.tr_Alley(X)
        X = self.tr_LandCont(X)
        X = self.tr_BldgType(X)
        X = self.tr_MasVnrType(X)
        X = self.tr_HouseStyle(X)
        X = self.tr_Neighborhood(X)
        X = self.tr_MSSubClass(X)
        return X

class drop_columns(BaseEstimator, TransformerMixin):
    '''
    Drops columns that are not useful for the model
    '''
    def __init__(self):
        self.columns = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        to_drop = [col for col in X.columns if 'NoGrg' in col]  # dropping dummies that are redundant
        to_drop += [col for col in X.columns if 'NoBsmt' in col]
        # other not useful columns
        to_drop += [col for col in X.columns if 'MSSubClass' in col]
        to_drop += [col for col in X.columns if 'Neighborhood' in col]  # maybe useful in the future
        to_drop += [col for col in X.columns if 'Condition1' in col]
        to_drop += [col for col in X.columns if 'Condition2' in col]
        to_drop += [col for col in X.columns if 'ExterCond' in col]  # maybe make it ordinal
        to_drop += [col for col in X.columns if 'Exterior1st' in col]
        to_drop += [col for col in X.columns if 'Exterior2nd' in col]
        to_drop += [col for col in X.columns if 'Functional' in col]
        to_drop += [col for col in X.columns if 'Heating_' in col]  # we don't want to drop the dummies of HeatingQC too
        to_drop += [col for col in X.columns if 'PoolQC' in col]
        to_drop += [col for col in X.columns if 'RoofMatl' in col]
        to_drop += [col for col in X.columns if 'RoofStyle' in col]
        to_drop += [col for col in X.columns if 'SaleCondition' in col]
        to_drop += [col for col in X.columns if 'SaleType' in col]
        to_drop += [col for col in X.columns if 'Utilities' in col]
        to_drop += [col for col in X.columns if 'BsmtCond' in col]  # maybe ordinal
        to_drop += [col for col in X.columns if 'Electrical' in col]
        to_drop += [col for col in X.columns if 'Foundation' in col]
        to_drop += [col for col in X.columns if 'LandSlope' in col]
        to_drop += [col for col in X.columns if 'Street' in col]
        to_drop += [col for col in X.columns if 'Fence' in col]
        to_drop += [col for col in X.columns if 'PavedDrive' in col]

        for col in to_drop:
            try:
                del X[col]
            except KeyError:
                pass
            
        self.columns = X.columns
        return X
    
    def get_feature_names(self):
        return list(self.columns)

class FeatureUnion_df(TransformerMixin, BaseEstimator):
    '''
    Wrapper of FeatureUnion but returning a Dataframe, 
    the column order follows the concatenation done by FeatureUnion

    transformer_list: list of Pipelines

    '''
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose  # these are necessary to work inside of GridSearch or similar
        self.feat_un = FeatureUnion(self.transformer_list, 
                                    self.n_jobs, 
                                    self.transformer_weights, 
                                    self.verbose)
        
    def fit(self, X, y=None):
        self.feat_un.fit(X)
        return self

    def transform(self, X, y=None):
        X_tr = self.feat_un.transform(X)
        columns = []
        
        for trsnf in self.transformer_list:
            cols = trsnf[1].steps[-1][1].get_features_name()  # getting the features name from the last step of each pipeline
            columns += list(cols)

        X_tr = pd.DataFrame(X_tr, index=X.index, columns=columns)
        
        return X_tr

    def get_params(self, deep=True):  # necessary to well behave in GridSearch
        return self.feat_un.get_params(deep=deep)

