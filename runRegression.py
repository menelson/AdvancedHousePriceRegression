'''
	An implementation of a housing regression analysis, including
	feature selection and hyperparameter tuning for a variety of
	different cases
'''

import pandas as pd
import matplotlib.pylab as plt
import os
import numpy as np
import cleanerClass as cc
import transformClass as tc
from plotFunctions import *
from trainTestFunctions import *

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor 
import xgboost as xgb
#import lightgbm as lgb

def initialisePipeline():
	''' 
		Let's initialise the pipeline in a separate method.
		Purely for clarity
	'''
	numeric_pipe = Pipeline([('fs', tc.feat_sel('numeric')),
                         ('imputer', tc.df_imputer(strategy='median')),
                         ('transf', tc.tr_numeric())])


	cat_pipe = Pipeline([('fs', tc.feat_sel('category')),
                     ('imputer', tc.df_imputer(strategy='most_frequent')), 
                     ('ord', tc.make_ordinal(['BsmtQual', 'KitchenQual','GarageQual',
                                           'GarageCond', 'ExterQual', 'HeatingQC'])), 
                     ('recode', tc.recode_cat()), 
                     ('dummies', tc.dummify())])


	processing_pipe = tc.FeatureUnion_df(transformer_list=[('cat_pipe', cat_pipe),
                                                 ('num_pipe', numeric_pipe)])


	full_pipe = Pipeline([('gen_cl', cc.general_cleaner()), 
                      ('processing', processing_pipe), 
                      ('scaler', tc.df_scaler()), 
                      ('dropper', drop_columns())])

	return full_pipe # Return the final combined pipeline

def main(plotDump=False, correlations=False, studyCats=False):
	'''
		First get the data and split it accordingly
	'''
	df_train = pd.read_csv('./data/train.csv') 
	df_test = pd.read_csv('./data/test.csv')

	print 'First twenty entries of training set: ', df_train.head(20)
	print 'Shape of the training set: ', df_train.shape #(Entries, features)

	# Make plots of the training features
	if plotDump:
		df_train.hist(bins=50, figsize=(20,15))
		plt.show()

	# We want to estimate the sale price, so treat this as the
	# target in the training set and then remove it. We'll take the
	# log to reduce skewness 
	df_train['target'] = np.log1p(df_train.SalePrice)
	del df_train['SalePrice']
	
	# Remove some outliers and implement a cleaning procedure
	df_train = df_train[df_train.GrLivArea < 4500].reset_index(drop=True)

	# We can now make our own test set out of the training set (this os effectively 
	# an additional validation set). The thing is, we need to split the training set
	# in such a way that preserves the structure of the data across a sensible
	# variable. We'll use Neighborhood with stratfied features
	train_set, test_set, train_target, test_target = train_test_split(df_train, df_train['target'], test_size=0.2, random_state=654, stratify=df_train.Neighborhood)
	
	# Apply the gc class to fix up the data frame structure
	gc = cc.general_cleaner()
	train_cleaned = train_set.copy()  # Copy since we will want to work on the fresh training set later
	#train_cleaned = gc.fit_transform(train_cleaned) # All of the initial cleaning has been moved to a pipeline
	print 'First twenty entries of cleaned training set: ', train_cleaned.head(20)
	print 'First twenty entries of training target: ', train_target.head(20)

	# Information the features (can be useful)
	features = train_cleaned.columns
	print 'The features in the cleaned training set are: ', features
	
	'''
		Use plotting macros to briefly study the data
	'''
	
	# Use the methods from plotFunctions for easy plotting  
	if correlations:
		high_corr = plot_correlations(train_cleaned, 'target', limit=20, annot = True)
		plot_distribution(train_cleaned, 'target')
		for col in high_corr[1:6].index:
    			plot_distribution(train_cleaned, col, correlation=high_corr)
		# Let's also try a bivariate analysis
		corr_target(train_cleaned, 'target', list(high_corr[1:12].index))
	
	# We want to shuffle through the various features to find the interesting ones, 
	# by e.g. accessing the similarity between the various feature distributions and
	# the target distribution. The below method does this, with a K-S test by
	# default. With can now study the categories associated to features of interest.
	if studyCats: 
		important_cats = find_cats(train_cleaned, 'target', thrs=0.3, critical=0.05)
		# Investigate some seemingly important features
		for cat in important_cats:
			segm_target(train_cleaned, cat, 'target')
			# Plot the category against the target, for different house styles (specified by the hue)
			plot_bivariate(train_cleaned, cat, 'target', hue='HouseStyle', alpha=0.7)
		plot_bivariate(train_cleaned, 'GrLivArea', 'target', hue='HouseStyle', alpha=0.7)
	plt.show()


	'''
		Let's now move on to models, validation, and error analysis
	'''
	full_pipe = initialisePipeline()
	full_pipe.fit_transfrom(train_cleaned)
	
	# Let's begin to study models using the pipeline
	models = [('lasso', Lasso(alpha=0.01)), ('ridge', Ridge()), ('sgd', SGDRegressor()), 
          ('forest', RandomForestRegressor(n_estimators=200)), ('xtree', ExtraTreesRegressor(n_estimators=200)), 
          ('svr', SVR()), 
          ('kneig', KNeighborsRegressor()),
          ('xgb', xgb.XGBRegressor(n_estimators=200, objective='reg:squarederror'))] 
          #('lgb', lgb.LGBMRegressor(n_estimators=200))]

	mod_name = []
	rmse_train = []
	rmse_test = []
	mae_train = []
	mae_test = []

	folds = KFold(5, shuffle=True, random_state=541)

	y = train_set['target'].copy()
	del train_set['target']
	y_test = test_set['target']
	del test_set['target']


	for model in models:
    		train = train_set.copy()
    		test = test_set.copy()
    		print model[0]
    		mod_name.append(model[0])
    
    		pipe = [('gen_cl', cc.general_cleaner()),
            		('processing', processing_pipe),
            		('scl', tc.df_scaler()),
            		('dropper', drop_columns())] + [model]
    
    		model_pipe = Pipeline(pipe)
            
    		inf_preds = cv_score(train, y, folds, model_pipe)
    
    		model_pipe.fit(train, y)  # refit on full train set
    
    		preds = model_pipe.predict(test)
    
    		rmse_train.append(mean_squared_error(y, inf_preds))
    		rmse_test.append(mean_squared_error(y_test, preds))
    		mae_train.append(mean_absolute_error(np.expm1(y), np.expm1(inf_preds)))
    		mae_test.append(mean_absolute_error(np.expm1(y_test), np.expm1(preds)))
    
    		print 'Train set RMSE: {0}'.format(round(np.sqrt(mean_squared_error(y, inf_preds)), 4))
    		print 'Train set MAE: {0}'.format(round(mean_absolute_error(np.expm1(y), np.expm1(inf_preds)), 2))
    		print 'Test set RMSE: {0}'.format(round(np.sqrt(mean_squared_error(y_test, preds)), 4))
    		print 'Test set MAE: {0}'.format(round(mean_absolute_error(np.expm1(y_test), np.expm1(preds)), 2))
    
    		print '_'*40
    		print '\n'
    
		results = pd.DataFrame({'model_name': mod_name, 
                        'rmse_train': rmse_train, 'rmse_test': rmse_test,
                        'mae_train': mae_train, 'mae_test': mae_test})	
	
	'''
		Perform hyperparameter tuning using a grid search, model by model
	'''

	# 1. Lasso regression hyperparameter tuning
	lasso_params = {'lasso__alpha': [0.002, 0.001, 0.0009, 0.0008], 
                'lasso__tol': [0.005, 0.001, 0.0005, 0.0001],
                'proc__cat__dummies__drop_first': [True, False],
                'proc__cat__ord__include_extra': ['include', 'dummies'], 
                'proc__num__transf__SF_room': [True, False],
                'proc__num__transf__bedroom': [True, False], 
                'proc__num__transf__lot': [True, False],
                'scaler__method': ['standard', 'robust']}

	result_lasso, bp_lasso, best_lasso = grid_search(train_set, y, lasso_pipe, 
                                                 param_grid=lasso_params, cv=folds, scoring='neg_mean_squared_error', 
                                                 random=100)

	# Get the best performance
	bp_lasso

	# 2. Ridge regression hyperparameter tuning 
	ridge_params = {'ridge__alpha': [2, 1.7, 1.5, 1.3, 1, 0.9], 
                'ridge__tol': [0.005, 0.001, 0.0005, 0.0001],
                'proc__cat__dummies__drop_first': [True, False],
                'proc__cat__ord__include_extra': ['include', 'dummies'], 
                'proc__num__transf__SF_room': [True, False],
                'proc__num__transf__lot': [True, False],
                'scaler__method': ['standard', 'robust']}

	result_ridge, bp_ridge, best_ridge = grid_search(train_set, y, ridge_pipe, 
                                                 param_grid=ridge_params, cv=folds, scoring='neg_mean_squared_error', 
                                                 random=100)

	bp_ridge

	# 3. Random forest hyperparameter tuning 
	forest_params = {'forest__max_depth': [10, 20, 30, None],
                 'forest__max_features': ['auto', 'sqrt', 'log2'], 
                 'forest__min_samples_leaf': [1, 3, 5, 10],
                 'forest__min_samples_split': [2, 4, 6, 8],
                'proc__cat__dummies__drop_first': [True, False],
                'proc__cat__ord__include_extra': ['include', 'dummies'], 
                'proc__num__transf__SF_room': [True, False], 
                'proc__num__transf__bath': [True, False], 
                'proc__num__transf__bedroom': [True, False], 
                'proc__num__transf__lot': [True, False], 
                'proc__num__transf__service': [True, False]}

	result_forest, bp_forest, best_forest = grid_search(train_set, y, forest_pipe, 
                                                 param_grid=forest_params, cv=folds, scoring='neg_mean_squared_error', 
                                                 random=100)

 	bp_forest

if __name__ == "__main__":

	from argparse import ArgumentParser
    	parser = ArgumentParser()
    	parser.add_argument("-d", "--plotDump", help="", action="store_true", default=False)
    	parser.add_argument("-c", "--correlations", help="", action="store_true", default=False)
    	parser.add_argument("-s", "--studyCats", help="", action="store_true", default=False)
	
	options = parser.parse_args()

	# Defining dictionary to be passed to the main function
    	option_dict = dict( (k, v) for k, v in vars(options).iteritems() if v is not None)
    	print option_dict
    	main(**option_dict)
