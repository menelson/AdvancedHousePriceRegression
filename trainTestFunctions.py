import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pylab as plt

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def cv_score(df_train, y_train, kfolds, pipeline, imp_coef=False):
    oof = np.zeros(len(df_train))
    train = df_train.copy()
    
    feat_df = pd.DataFrame()
    
    for n_fold, (train_index, test_index) in enumerate(kfolds.split(train.values)):
            
        trn_data = train.iloc[train_index][:]
        val_data = train.iloc[test_index][:]
        
        trn_target = y_train.iloc[train_index].values.ravel()
        val_target = y_train.iloc[test_index].values.ravel()
        
        pipeline.fit(trn_data, trn_target)

        oof[test_index] = pipeline.predict(val_data).ravel()

        if imp_coef:
            try:
                fold_df = get_coef(pipeline)
            except AttributeError:
                fold_df = get_feature_importance(pipeline)
                
            fold_df['fold'] = n_fold + 1
            feat_df = pd.concat([feat_df, fold_df], axis=0)
       
    if imp_coef:
        feat_df = feat_df.groupby('feat')['score'].agg(['mean', 'std'])
        feat_df['abs_sco'] = (abs(feat_df['mean']))
        feat_df = feat_df.sort_values(by=['abs_sco'],ascending=False)
        del feat_df['abs_sco']
        return oof, feat_df
    else:    
        return oof


def grid_search(data, target, estimator, param_grid, scoring, cv, random=False):
    
    if random:
        grid = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, cv=cv, scoring=scoring, 
                                  n_iter=random, n_jobs=-1, random_state=434, iid=False)
    else:
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                            cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    
    pd.options.mode.chained_assignment = None  # turn on and off a warning of pandas
    tmp = data.copy()
    grid = grid.fit(tmp, target)
    pd.options.mode.chained_assignment = 'warn'
    
    result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', 
                                                        ascending=False).reset_index()
    
    del result['params']
    times = [col for col in result.columns if col.endswith('_time')]
    params = [col for col in result.columns if col.startswith('param_')]
    
    result = result[params + ['mean_test_score', 'std_test_score'] + times]
    
    return result, grid.best_params_, grid.best_estimator_


def get_coef(pipe):
    imp = pipe.steps[-1][1].coef_.tolist()
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result['abs_res'] = abs(result['score'])
    result = result.sort_values(by=['abs_res'],ascending=False)
    del result['abs_res']
    return result


def get_feature_importance(pipe):
    imp = pipe.steps[-1][1].feature_importances_.tolist() #it's a pipeline
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result


def _plot_diagonal(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = min(xmin, xmax)
    high = max(xmin, xmax)
    scl = (high - low) / 100
    
    line = pd.DataFrame({'x': np.arange(low, high ,scl), # small hack for a diagonal line
                         'y': np.arange(low, high ,scl)})
    ax.plot(line.x, line.y, color='black', linestyle='--')
    
    return ax


def plot_predictions(data, true_label, pred_label, feature=None, hue=None, legend=False):
    
    tmp = data.copy()
    tmp['Prediction'] = pred_label
    tmp['True Label'] = true_label
    tmp['Residual'] = tmp['True Label'] - tmp['Prediction']
    
    diag = False
    alpha = 0.7
    label = ''
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    
    if feature is None:
        feature = 'True Label'
        diag = True
    else:
        legend = 'full'
        sns.scatterplot(x=feature, y='True Label', data=tmp, ax=ax[0], label='True',
                         hue=hue, legend=legend, alpha=alpha)
        label = 'Predicted'
        alpha = 0.4

    sns.scatterplot(x=feature, y='Prediction', data=tmp, ax=ax[0], label=label,
                         hue=hue, legend=legend, alpha=alpha)
    if diag:
        ax[0] = _plot_diagonal(ax[0])
    
    sns.scatterplot(x=feature, y='Residual', data=tmp, ax=ax[1], 
                    hue=hue, legend=legend, alpha=0.7)
    ax[1].axhline(y=0, color='r', linestyle='--')
    
    ax[0].set_title('{0} vs Predictions'.format(feature))
    ax[1].set_title('{0} vs Residuals'.format(feature))
