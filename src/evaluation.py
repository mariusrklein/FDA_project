from typing import Dict, Tuple, Callable
import os
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy.builtins import Q
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse import csc_matrix, csr_matrix
from itertools import chain
import anndata as ad
from src.correction import normalize_proportion_ratios
from src import const
import seaborn as sns
import scanpy as sc
import re
from sklearn.svm import LinearSVC



## ------------------------ ##
##   EVALUATION FUNCTIONS   ##
## ------------------------ ##


def plot_all_ion_slopes(
    am_adata, 
    subset = [], 
    col_wrap = 10, 
    ratios = True,
    normalized = True,
    log = True,
    kde = False):

    norm_int = am_adata.copy()
    if ratios:
        norm_int = normalize_proportion_ratios(am_adata, normalized=normalized)


    ions = [] 
    if len(subset) == 0:
        ions = list(am_adata.var_names)
    else:
        ions = list(subset)
    
    ions.append(const.TPO)
    # print(ions)
    grid_slopes_df = sc.get.obs_df(norm_int, keys=ions)
    grid_slopes_df['am'] = grid_slopes_df.index
    plot_df = grid_slopes_df.melt(id_vars=['am', const.TPO], var_name='ion')
    if log:
        plot_df[[const.TPO, 'value']] = np.log10(plot_df[[const.TPO, 'value']])
    grid = sns.FacetGrid(plot_df, col = 'ion', col_wrap = col_wrap)
    if kde:
        grid.map(sns.kdeplot, const.TPO, 'value', fill=True)
    else:
        grid.map(sns.scatterplot, const.TPO, 'value')
    for i, ax in enumerate(grid.axes.flat): 
        ax.axline((0,0), slope=-1)
        ax.set(ylabel='log intensity / sampling prop. ratio', xlabel = 'log sampling proportion')

        
    
def compare_pre_post_correction(adata, adata_cor, proportion_threshold, row='well', ions=None, wells=None, ratio=True, normalized=True):
    
    if wells is None:
        wells = list(set(adata.obs[row]))[:3]
        
    if ions is None:
        
        if ratio:
            adata = normalize_proportion_ratios(intensities_ad=adata, normalized=normalized)

        global_df = sc.get.obs_df(adata_cor, keys=[row, const.TPO] + list(adata_cor.var_names)
                                 ).melt(id_vars=[row, const.TPO], var_name='ion', value_name='intensity')
        
        if ratio:
            global_df['intensity'] = np.log10(1+global_df['intensity'])
            global_df[const.TPO] = np.log10(1+global_df[const.TPO])
            
        avail_ions = list(set(global_df[(global_df['intensity']>0) & 
                                        (global_df[const.TPO]>0) & (global_df[row].isin(wells))]['ion'] ))
      #  print(avail_ions)
        var_table = adata_cor.var[(adata_cor.var.index.isin(avail_ions)) & (adata_cor.var['correction_n_datapoints']>50)
                                 ].sort_values('correction_quantreg_slope')
        
        ions = list(var_table.head(2).index) + list(var_table.sort_values('correction_n_datapoints').tail(2).index) + list(var_table.tail(2).index)
#        ions = list(var_table.sort_values('correction_n_datapoints').tail(2).index) + list(var_table.tail(2).index)
      
#    print(adata_cor.var[adata_cor.var.index.isin(ions)])
    
    def plot_all_wells(adata, ions, wells, row='well', x=const.TPO, ratio=True, normalized=True, proportion_threshold = 0.1):

        adata = adata[adata.obs[row].isin(wells)]
        yscale = 'intensity'
        
        if ratio:
            adata = normalize_proportion_ratios(intensities_ad=adata, normalized=normalized)
            yscale = 'intensity_proportion_ratio'

        plot_df = sc.get.obs_df(adata, keys=[row, x] + ions).melt(id_vars=[row, x], var_name='ion', value_name=yscale)
        
        # getting rid of NaNs and -Inf values
        plot_df = plot_df[plot_df[yscale] > 0]
        
        # mark points that were used for quantreg correction 
        plot_df['include_quantreg'] = plot_df[x] > proportion_threshold
        if ratio:
            plot_df[yscale] = np.log10(plot_df[yscale])
            plot_df[x] = np.log10(plot_df[x])

        binh = np.ptp(plot_df[yscale]) / 30
        binw = np.ptp(plot_df[x]) / 30
        
        graph = sns.FacetGrid(plot_df, row=row, col='ion', hue='include_quantreg', sharey=False, margin_titles=True, palette={True: 'tab:green', False: 'tab:red'})
        # graph.map(sns.histplot, x, yscale, stat='frequency', binwidth = (binw, binh)).add_legend(title='Used for correction')
        graph.map(sns.scatterplot, x, yscale, size=1).add_legend(title='Used for correction')
        
        if ratio:
            graph.set_axis_labels(const.LABEL['SProp'], const.LABEL['IRatio'])
        else:
            graph.set_axis_labels(const.LABEL['SProp'], 'MALDI intensity')
        # graph.set(ylim=(-1.2, 3.2))
        # graph.set(xlim=(-3.2, 0.2))
        
        params = []

        for well_i, well in enumerate(wells):
            for ion_i, i in enumerate(ions):
                q_df = plot_df[(plot_df['ion'] == i) & (plot_df[row] == well)]

                if len(q_df) == 0:
                    # params[ion_i] = {'Intercept': np.nan, x: np.nan}
                    continue
                model = smf.quantreg(yscale+' ~ '+x, q_df)
                qrmodel = model.fit(q=0.5)
                param_dict = {'ion': i, row: well, 'intercept_glob': qrmodel.params[0], 'slope_glob': qrmodel.params[1]}
                #print(graph.axes)
                graph.axes[well_i][ion_i].axline((0, param_dict['intercept_glob']), slope=param_dict['slope_glob'], color='black')

                if 'correction_quantreg_slope' not in adata.var.columns and ratio:
                    q_df = plot_df[(plot_df['ion'] == i) & (plot_df[row] == well) & (plot_df['include_quantreg'] == True)]
                    model = smf.quantreg(yscale+' ~ '+x, q_df)
                    qrmodel = model.fit(q=0.5)
                    graph.axes[well_i][ion_i].axline((0, qrmodel.params[0]), slope=qrmodel.params[1], color='red')
                    param_dict['slope_correction'] = qrmodel.params[1]
                    param_dict['intercept_correction'] = qrmodel.params[0]
                    #orig_params = adata_cor.var.loc[i, ['correction_quantreg_slope', 'correction_quantreg_intersect']]
                    #graph.axes[well_i][ion_i].axline((0, orig_params['correction_quantreg_intersect']), slope=orig_params['correction_quantreg_slope'], color='red')
                    #param_dict['slope_correction'] = orig_params['correction_quantreg_slope']
                    #param_dict['intercept_correction'] = orig_params['correction_quantreg_intersect']
                params.append(param_dict)


        return pd.DataFrame(params).sort_values(['ion', row]).set_index(['ion', row])
        
        
    df1 = plot_all_wells(adata, ions=ions, wells=wells, ratio=ratio, normalized=normalized, proportion_threshold=proportion_threshold)
    df2 = plot_all_wells(adata_cor, ions=ions, wells=wells, ratio=ratio, normalized=normalized, proportion_threshold=proportion_threshold)
    out_df = pd.merge(df1, df2, right_index=True, left_index=True, suffixes=('_uncorrected', '_ISM_correction')).loc[ions]
    return out_df[sorted(out_df.columns)]


def plot_deviations_between_adatas(adata, gen_adata, adata_cor, well_name = 'well'):

    def plot_deviations(adata1, adata2, label=""):
        df = (adata2.to_df() - adata1.to_df()) / adata1.to_df()
        # df = df / (adata1.to_df().shape[0] * adata1.to_df().shape[1])
        df = df.mean(axis=1)

        df = pd.concat({'mean relative deviation': df, well_name: adata1.obs[well_name]}, axis=1)
        first = df.groupby("well").quantile(q=[0.25]).reset_index()[[well_name, 'mean relative deviation']].set_index(well_name)
        third = df.groupby("well").quantile(q=[0.75]).reset_index()[[well_name, 'mean relative deviation']].set_index(well_name)

        plt = sns.lineplot(df, x=well_name, y="mean relative deviation", label=label, marker='o', errorbar=None )
        plt.fill_between(first.index, first['mean relative deviation'], third['mean relative deviation'], alpha=0.2)
        plt.set(title = 'Summed absolute deviations across wells', ylabel=const.LABEL['RDev_general'])
        plt.set_xticklabels(plt.get_xticklabels(), rotation=45, horizontalalignment='right')
    
        df_out = pd.concat([df.groupby("well").median(), third-first], axis=1)
        df_out.columns = [label+"_median", label+"_iqr"]
        return df_out

    df1 = plot_deviations(adata, gen_adata, 'gen. data vs. spacem')
    df2 = plot_deviations(gen_adata, adata_cor, 'corr. data vs. gen. data')
    
    return pd.concat([df1, df2], axis=1)




class MetaboliteAnalysis:
    
    def __init__(self, 
                 adata, 
                 adata_cor, 
                 condition_name,
                 well_name = "well",
                 comparison_name = 'correction',
                 obs_columns = [],
                 var_columns = [],
                 use_raw = False,
                 exclude_pool_corrected = False,
                 p_val_threshold = 0.001,
                 q_val_threshold = 0.05,
                 de_score_threshold = -2,
                 
                ):
        self.adata = adata
        self.adata_cor = adata_cor
        self.obs_columns = obs_columns
        
        has_correction = 'corrected_only_using_pool' in var_columns
        
        self.conc_adata = ad.concat({'uncorrected': adata, 'ISM correction': adata_cor}, 
                                    label='correction', index_unique='_', merge='same')
        
        sc.tl.rank_genes_groups(adata=self.conc_adata, groupby='correction', use_raw=use_raw, reference='uncorrected', method='wilcoxon')
        self.all_ions_dea = sc.get.rank_genes_groups_df(self.conc_adata, group='ISM correction').set_index('names')

        self.impact_ions = pd.merge(self.all_ions_dea, self.adata_cor.var[var_columns], 
                                    how='left', left_index=True, right_index=True)
        self.included_molecules = list(self.impact_ions.index)
        
        self.impact_ions = self.impact_ions.sort_values(by='scores')
        
        self.impact_ions['logfoldchanges'] = self.impact_ions['logfoldchanges'].replace(-np.Inf, min(self.impact_ions['logfoldchanges'].replace(-np.Inf, 0))*1.1)
        self.impact_ions['logfoldchanges'] = self.impact_ions['logfoldchanges'].replace(np.nan, min(self.impact_ions['logfoldchanges'].replace(-np.nan, 0))*1.1)
        
        self.impact_ions['significant'] = (self.impact_ions['pvals'] < p_val_threshold) & (self.impact_ions['pvals_adj'] < q_val_threshold)
        self.sign_impact_ions = self.impact_ions[self.impact_ions['significant'] == True]
        
        if has_correction:
            self.impact_ions_filtered = self.impact_ions[self.impact_ions['corrected_only_using_pool'] == False]
            self.included_molecules_filtered = list(self.impact_ions[self.impact_ions['corrected_only_using_pool'] == False].index)
        
        self.conc_adata_raw = self.conc_adata.copy()
        self.conc_adata_raw.obs['cell'] = [re.sub('_[a-zA-Z ]+$', '', i) for i in self.conc_adata_raw.obs.index]
        if use_raw:
            self.conc_adata_raw.X = self.conc_adata_raw.raw.X
        
        self.obs_columns.extend([comparison_name, condition_name, 'cell', well_name])
        self.changed_ions_df = sc.get.obs_df(self.conc_adata_raw, 
                                        keys=(self.obs_columns+list(self.conc_adata.var_names))).melt(id_vars=self.obs_columns, 
                                                                                                 var_name='ion')
        self.obs_columns.remove(comparison_name)
        self.obs_columns.append('ion')
        self.plot_df = self.changed_ions_df.pivot(index=self.obs_columns, columns=comparison_name, values='value')
        self.plot_df.reset_index(inplace=True)
        self.plot_df['quotient'] = self.plot_df['ISM correction'] / self.plot_df['uncorrected'].replace(0, 1)
        
        if has_correction:
            self.plot_df_filtered = self.plot_df[self.plot_df['ion'].isin(self.included_molecules_filtered)]
    
        
    def summary(self):
        print('Of %1d ions, %1d are significantly altered by ion suppression correction (p < 1e-3, DEA score < -2).'
              %(len(self.impact_ions), len(self.sign_impact_ions)))

        print('Association of significance and pool correction:')
        sign = pd.Series(['significant' if a else 'not significant' 
            for a in self.impact_ions['significant']], name='Effect on molecule')
        ref_pool = pd.Series(['corrected using ref. data' if a else 'corrected using own data' 
            for a in self.impact_ions['corrected_only_using_pool']], name='Mode of correction')
        print(pd.crosstab(sign, ref_pool, margins=True))
        
    def pair_plot(self, exclude_ref_corrected=True, **kwargs):
        
        impact_ions = self.impact_ions_filtered if exclude_ref_corrected else self.impact_ions
            
        def correlate_ions(ion):
            return self.plot_df[self.plot_df.ion == ion].corr(numeric_only = True)['uncorrected']['ISM correction']
        
        # if 'pearson' not in self.impact_ions.columns:
        #     impact_ions['pearson'] = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(correlate_ions)(ion) for ion in tqdm(impact_ions.index))
        

        sns.pairplot(impact_ions[['logfoldchanges', 'mean_correction_quantreg_slope', 
                                  'median_intensity', 'mean_intensity',
                                  'corrected_only_using_pool', 'n_cells']], 
                     hue='corrected_only_using_pool',
                     **kwargs
                    )
            
    def volcano_plot(self, exclude_ref_corrected = True, **kwargs):

        volcano_df = self.impact_ions_filtered if exclude_ref_corrected else self.impact_ions
        volcano_df['-log pval'] = -np.log10(volcano_df['pvals'])
        fig, ax = plt.subplots(1,2)
        sns.scatterplot(volcano_df, x='logfoldchanges', y='-log pval', ax=ax[0], hue='significant', **kwargs)
        ax[0].set_title('Intensity LFC by ion suppression correction')
        sns.scatterplot(volcano_df, x='logfoldchanges', y='-log pval', ax=ax[1], hue='corrected_only_using_pool', **kwargs)
        ax[1].set_title('Intensity LFC by ion suppression correction')
        fig.set_figwidth(12)
    
    def top_ion_plot(self, top_ions = 5, plot_method=sns.histplot, exclude_ref_corrected = True, log=False, **kwargs):
        
        plot_df = self.plot_df_filtered if exclude_ref_corrected else self.plot_df
        plot_df = plot_df[plot_df.uncorrected > 0]
        impact_ions = self.impact_ions_filtered if exclude_ref_corrected else self.impact_ions
        
        tops = {
            'high DE score': list(impact_ions.head(top_ions).index),
            'low DE score': list(impact_ions.tail(top_ions).index),
            'high slope': list(impact_ions.sort_values(by='mean_correction_quantreg_slope').head(top_ions).index),
            'low slope': list(impact_ions.sort_values(by='mean_correction_quantreg_slope').tail(top_ions).index),
        }
        
        for condition, ion_list in tops.items():
            self.ion_plot(ions=ion_list, title=condition, plot_method=plot_method, 
                          exclude_ref_corrected=exclude_ref_corrected, log=log, **kwargs)
        
    def ion_plot(self, ions, title='', plot_method=sns.histplot, exclude_ref_corrected=True, log=False, **kwargs):

        plot_df = self.plot_df_filtered.copy() if exclude_ref_corrected else self.plot_df.copy()
        plot_df = plot_df[self.plot_df.uncorrected > 0]
        impact_ions = self.impact_ions_filtered if exclude_ref_corrected else self.impact_ions
        
        plot_df['set'] = title
        
        params = {
            'quantreg_slope': [impact_ions.loc[ion, 'mean_correction_quantreg_slope'] for ion in ions],
           # 'pearson': [plot_df[plot_df.ion == ion].corr(numeric_only=True)['uncorrected']['ISM correction'] for ion in ions],
            'DE scores': [float(impact_ions.loc[ion, 'scores']) for ion in ions],
            'corrected_using_pool': [float(impact_ions.loc[ion, 'sum_correction_using_ion_pool']) for ion in ions],
           # 'n_cells': [float(impact_ions.loc[ion, 'n_cells']) for ion in ions],
        }
        grid = sns.FacetGrid(plot_df, col='ion', row='set', sharex=False, sharey=False, col_order=ions, palette='cividis', margin_titles=True)
        grid.map(plot_method, 'uncorrected', 'ISM correction', **kwargs).add_legend()
        if not log:
            grid.set(aspect = 1)
        for i, ax in enumerate(grid.axes.flat): 
            lim_max = max([ax.get_xlim()[1], ax.get_ylim()[1]])
            lim_min = min([ax.get_xlim()[0], ax.get_ylim()[0]])
            
            textstr = '\n'.join(['%s = %1.3f'%(name, value[i]) for name, value in params.items()])
            #  'quantreg slope = %1.3f\npearson r = %1.5f\nscores = %1.2f'%(slopes[i], pearson[i], scores[i])
            props = dict(boxstyle='round', alpha=0.5)
            
            if log:
                #ax.set_xlim(np.log(lim_min), np.log(lim_max))
                #ax.set_ylim(np.log(lim_min), np.log(lim_max))
                
                ax.axline((1,1), (2,2))
            else:
                ax.set_xlim(lim_min, lim_max)
                ax.set_ylim(lim_min, lim_max)
               
                ax.axline((lim_min,lim_min), slope=1)
                
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props, fontsize=10)
    
    def quotient_plot(self, show_cells = None, show_TPO = True):
        
        self.long_plot_df = self.plot_df.melt(id_vars=self.obs_columns, var_name='correction')
        self.long_plot_df = pd.merge(self.long_plot_df, self.adata_cor.var[['corrected_only_using_pool']], left_on='ion', right_index=True)

        self.cells = list(set(self.long_plot_df['cell']))[:6] if show_cells is None else show_cells
        cells_df = pd.DataFrame(index=self.cells)
        cells_df['set_TPO'] = [self.adata_cor.obs.loc[c, 'list_TPO'] for c in self.cells]
        cells_df['list_TPO'] = [list(np.float_(row.split(";"))) for row in cells_df['set_TPO']]

        grid = sns.FacetGrid(self.long_plot_df[self.long_plot_df['cell'].isin(self.cells)], row='correction', col='cell', 
                             hue='corrected_only_using_pool', margin_titles=True, sharey=False, col_order = self.cells)
        grid.map(sns.lineplot, 'ion', 'value', linewidth = 0.5).add_legend()
        for i, ax in enumerate(grid.axes.flat): 
            ax.set_xticks([])
            if i >= 2*len(self.cells):
            #if i % 3 == 2:
                #cell = self.cells[round(i/3-1/3)]
                cell = self.cells[round(i-2*len(self.cells))]
                ax.text(0.05, 0.95, cell, transform=ax.transAxes, verticalalignment='top', fontsize=10)
                if show_TPO:
                    for line in cells_df.loc[cell, 'list_TPO']:
                        ax.axhline(y=line, color='black')
                        
                        
        if show_TPO:
            cell_impact_ions = self.long_plot_df.loc[(self.long_plot_df['cell'].isin(self.cells)) & 
                                                     (self.long_plot_df['correction'] == 'quotient') & 
                                                     (self.long_plot_df['value'] > 0)]
            grid = sns.FacetGrid(cell_impact_ions, col='cell', row='corrected_only_using_pool', 
                                 col_order=self.cells, hue='corrected_only_using_pool', sharey=False, 
                                 margin_titles=True)
            grid.map(sns.kdeplot, 'value').add_legend()
            for i, ax in enumerate(grid.axes.flat): 
                cell = self.cells[i % len(self.cells)]
                # cell = re.sub('cell = ', '', ax.get_title())
                for line in cells_df.loc[cell, 'list_TPO']:
                    ax.axvline(x=line, color='black')
                # ax.text(0.05, 0.95, cell, transform=ax.transAxes, verticalalignment='top', fontsize=10)
                ax.set_xlabel('corrected / raw ion intensities ratio')
            

    def save_matrix(self, save_to_path, safe_to_name = 'metenrichr', save_figures = None):
        subset_adata = self.conc_adata_raw.copy()
        df = subset_adata.to_df()
        # compress data by rounding
        significant_figures = int(df.shape[0] * df.shape[1] / 6e6) if save_figures is None else save_figures
        df = df.applymap(lambda x: 0 if x == 0 else round(x, significant_figures - int(np.floor(np.log10(np.abs(x))))))
        df.to_csv(os.path.join(save_to_path, safe_to_name+'_matrix.csv'))

        metadata = pd.DataFrame(subset_adata.obs['correction'])
        metadata['correction'].name = 'condition'
        metadata.to_csv(os.path.join(save_to_path, safe_to_name+'_metadata.csv'))
        
        return (df, metadata)

        
        

def analyse_corrected_metabolites(adata,
    adata_cor,
    condition_name,
    top_ions = 5,
    save_to_path = None,
    exclude_pool_corrected = True,
    volcano_plot = True,
    pair_plot = True,
    top_plot = True
):
    conc_adata = ad.concat({'uncorrected': adata, 'ISM correction': adata_cor}, label='correction', index_unique='_', merge='same')
    
    sc.tl.rank_genes_groups(adata=conc_adata, groupby='correction', use_raw=True, method='wilcoxon')
    all_ions_dea = sc.get.rank_genes_groups_df(conc_adata, group='ISM correction')

    adata_cor.var['names'] = adata_cor.var.index
    impact_ions = pd.merge(all_ions_dea, adata_cor.var[['names', 'correction_using_ion_pool', 'correction_n_iterations', 'correction_n_datapoints', 'mean_correction_quantreg_slope']], on='names', how='left')
    impact_ions = impact_ions.sort_values(by='scores')
    impact_ions['logfoldchanges'] = impact_ions['logfoldchanges'].replace(-np.Inf, min(impact_ions['logfoldchanges'])*1.1)

    impact_ions['significant'] = (impact_ions['pvals'] < 0.001) & (impact_ions['scores'] < -2)
    sign_impact_ions = impact_ions[impact_ions['significant'] == True]
    print('Of %1d ions, %1d are significantly altered by ion suppression correction (p < 1e-3, DEA score < -2).'%(len(impact_ions), len(sign_impact_ions)))
    
    print('Association of significance and pool correction')
    print(pd.crosstab(impact_ions['significant'], impact_ions['correction_using_ion_pool'], margins=True))
    included_molecules = list(impact_ions['names'])
    if exclude_pool_corrected:
        print('excluding pool_corrected')
        impact_ions = impact_ions[impact_ions['correction_using_ion_pool'] == False]
        included_molecules = list(impact_ions.loc[impact_ions['correction_using_ion_pool'] == False, 'names'])
    
    # impact_ions
    if volcano_plot:
        print('volcano plot of changed intensities')
        volcano_df = impact_ions
        volcano_df['-log pval'] = -np.log10(volcano_df['pvals'])
        fig, ax = plt.subplots(1,2)
        sns.scatterplot(volcano_df, x='logfoldchanges', y='-log pval', ax=ax[0], hue='significant')
        sns.scatterplot(volcano_df, x='scores', y='-log pval', ax=ax[1], hue='significant')
        ax[0].set_title('Intensity LFC by ion suppression correction')
        ax[1].set_title('Corresponding scores')
        fig.set_figwidth(12)
    
    conc_adata_raw = conc_adata.copy()
    conc_adata_raw.obs['cell'] = [re.sub('_[a-zA-Z ]+$', '', i) for i in conc_adata_raw.obs.index]
    conc_adata_raw.X = conc_adata_raw.raw.X
    changed_ions_df = sc.get.obs_df(conc_adata_raw, keys=(['correction', condition_name, 'cell', 'well']+list(conc_adata.var_names)))
    plot_df = changed_ions_df.melt(id_vars=['correction', condition_name, 'cell', 'well'], var_name='ion').pivot(index=['ion', condition_name, 'cell', 'well'], columns='correction', values='value')
    plot_df.reset_index(inplace=True)
    plot_df = plot_df[plot_df.uncorrected > 0]
    
    # impact_ions
    if pair_plot:
        def correlate_ions(ion):
            return plot_df[plot_df.ion == ion].corr(numeric_only = True)['uncorrected']['ISM correction']
        print('associations between measures')
        impact_ions['pearson'] = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(correlate_ions)(ion) for ion in tqdm(impact_ions['names']))
        sns.pairplot(impact_ions[['scores', 'logfoldchanges', 'mean_correction_quantreg_slope', 'pearson', 
                                  'significant']], 
                     hue='significant'
                    )
        if exclude_pool_corrected == False:
            sns.pairplot(impact_ions[['scores', 'logfoldchanges', 'mean_correction_quantreg_slope', 'pearson', 
                                      'mean_correction_quantreg_slope']], 
                         hue='mean_correction_quantreg_slope'
                        )
    # plot_df, adata_cor
    if top_plot:
        print('top most and least corrected ions')
        ions_corr = list(impact_ions['names'].head(top_ions))
        ions_uncorr = list(impact_ions['names'].tail(top_ions))
        ions_max_slope = list(impact_ions.sort_values(by='mean_correction_quantreg_slope')['names'].head(top_ions))
        ions_min_slope = list(impact_ions.sort_values(by='mean_correction_quantreg_slope')['names'].tail(top_ions))
        ions = ions_corr + ions_uncorr + ions_max_slope + ions_min_slope
        # simple way to remove duplicates
        ions = list(dict.fromkeys(ions))
        plot_df['group'] = ['strongly corr' if (ion in ions_corr) 
                            else 'hardly corr' if (ion in ions_uncorr)
                            else 'hardly corr' if (ion in ions_uncorr)
                            else 'max slope' if (ion in ions_max_slope)
                            else 'min_slope' if (ion in ions_min_slope)
                            else 'other' for ion in plot_df['ion'] ]

        slopes = [adata_cor.var.loc[ion, 'correction_quantreg_slope'] for ion in ions]
        # intersects = [adata_cor.var.loc[ion, 'correction_quantreg_intersect'] for ion in ions]
        pearson = [plot_df[plot_df.ion == ion].corr(numeric_only=True)['uncorrected']['ISM correction'] for ion in ions]
        scores = [float(impact_ions.loc[impact_ions.names == ion, 'scores']) for ion in ions]
        grid = sns.FacetGrid(plot_df, col='ion', hue='group', col_wrap=5, sharex=False, sharey=False, col_order=ions, palette='cividis')
        grid.map(sns.histplot, 'uncorrected', 'ISM correction').add_legend()
        grid.set(aspect = 1)
        for i, ax in enumerate(grid.axes.flat): 
            lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
            lim_min = min([ax.get_xlim()[1], ax.get_ylim()[1]])
            textstr = 'quantreg slope = %1.3f\npearson r = %1.5f\nscores = %1.2f'%(slopes[i], pearson[i], scores[i])
            props = dict(boxstyle='round', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props, fontsize=10)
            ax.axline((lim_min,lim_min), slope=1)
            
    if save_to_path is not None:
        subset_adata = conc_adata_raw.copy()
        df = subset_adata.to_df()
        # compress data by rounding
        significant_figures = int(df.shape[0] * df.shape[1] / 6e6)
        df = df.applymap(lambda x: 0 if x == 0 else round(x, significant_figures - int(np.floor(np.log10(np.abs(x))))))
        df.to_csv(os.path.join(save_to_path, 'metenrichr_matrix.csv'))

        metadata = pd.DataFrame(subset_adata.obs['correction'])
        metadata['correction'].name = 'condition'
        metadata.to_csv(os.path.join(save_to_path, 'metenrichr_metadata.csv'))
    
    return (impact_ions, plot_df)
        
    


def plot_intermixing_graph(ax, summary_df, label):
    ax.plot(summary_df.index, summary_df['mean'], label=label)
    # ax.fill_between(summary_df.index, (summary_df['mean'] - summary_df['sd']), summary_df['mean'] + summary_df['sd'], alpha=0.2)
    # ax.set_ylim(0, 1.05)
    # ax.set_xscale('log')
    ax.set_xlabel('Relative neighborhood size')
    ax.set_ylabel('Cluster intermixing')

def intermixing_metric_sampled(
    adata_orig: ad.AnnData,
    condition_name: str,
    sample_frac=0.2,
    ax=None,
    label="",
    measure = 'X_umap',
    n_datapoints = 100,
    sample_log = False,
    neighborhood_size = 100,
    normalized = False,
    n_jobs = multiprocessing.cpu_count()
):

    if neighborhood_size is None:
        neighborhood_size = len(adata_orig.obs)
    if n_datapoints is None:
        n_datapoints = neighborhood_size
        
    sample = adata_orig.obs.groupby(condition_name,
        group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=1))

    adata = adata_orig.copy()
    adata = adata[sample.index]

    dist_matrix: np.ndarray
    if measure == "X":
        dist_matrix = distance_matrix(adata.X,
            adata_orig.X)
    else:
        dist_matrix = distance_matrix(adata.obsm[measure],
            adata_orig.obsm[measure])
    
    
    if n_datapoints == 1:
        sampling_range = list(chain(*[neighborhood_size]))
    else:
        if sample_log:
            sampling_range = np.unique(np.logspace(0, np.log10(neighborhood_size), n_datapoints).astype(int))
        else:
            sampling_range = range(1, neighborhood_size, 
                round(neighborhood_size / n_datapoints))
    #print(sampling_range)
    norm_factors = ( adata.obs[condition_name].value_counts() / len(adata.obs) 
        * len(adata.obs[condition_name].value_counts()) )

    #neighborhood_df = pd.DataFrame(columns=sampling_range, index=adata.obs['ObjectNumber'])

    def get_neighborhood_series(index, celltype):
        neighbors = pd.Series(dist_matrix[index],
            index=adata_orig.obs[condition_name]).sort_values()
        if normalized:
            return [ 1 - (neighbors[:i].index.value_counts()[celltype] / i / 
                norm_factors[celltype])  for i in sampling_range]
        return [ 1 - (neighbors[:i].index.value_counts()[celltype] / i) for i in sampling_range]

    neighborhood_df = pd.DataFrame(
        Parallel(n_jobs=n_jobs)(delayed(get_neighborhood_series)(index, celltype) 
        for index, celltype in enumerate(adata.obs[condition_name])),
        columns=sampling_range, index=adata.obs.index
    )
    
    summary = pd.concat([neighborhood_df.mean(axis=0), neighborhood_df.std(axis=0)], axis=1)
    summary.columns = ["mean", "sd"]
    summary['rel_neighborhood'] = np.linspace(0, 1, len(summary))
    if ax is not None:
        plot_intermixing_graph(ax, summary, label)
        if sample_log:
            ax.set_xscale('log')

    return (neighborhood_df, summary)


def intermixing(adata_dict: Dict[str, ad.AnnData],
    condition_name: str,
    measures = ['X', 'X_pca', 'X_umap'],
    show_table = [10],
    sample_frac = 0.2,
    sample_log = False,
    n_datapoints = 100,
    neighborhood_size = 100,
    normalized = True,
    n_jobs = multiprocessing.cpu_count()
)  -> pd.DataFrame:

    fig, ax = plt.subplots(1, len(measures), sharey=True)
    if len(measures) == 1:
        ax = [ax]
        
    fig.set_figwidth(4*len(measures))
    
    summaries = {}
    results = {}

    for i, measure in enumerate(measures):
        print(measure)
        for label, adata in adata_dict.items():
            res = intermixing_metric_sampled(adata_orig = adata, 
                condition_name = condition_name, 
                sample_frac = sample_frac, 
                sample_log = sample_log,
                ax = ax[i], 
                label = measure+'_'+label, 
                measure = measure,
                n_datapoints = n_datapoints,
                neighborhood_size = neighborhood_size,
                normalized = normalized,
                n_jobs = n_jobs
            )
            
            summaries[measure+'_'+label] = res[1]
            results[measure+'_'+label] = res[0]

        ax[i].legend()



    fig.tight_layout()
    
    if not all([hood in list(summaries.values())[0].index for hood in show_table]):
        summaries = {}
        for i, measure in enumerate(measures):
            for label, adata in adata_dict.items():
                summaries[measure+'_'+label] = intermixing_metric_sampled(adata_orig = adata, 
                    condition_name = condition_name,
                    neighborhood_size = show_table,
                    sample_frac = sample_frac,
                    label = measure+'_'+label,
                    measure = measure,
                    normalized = True,
                    n_jobs = n_jobs
                )

    print(pd.concat({k: v.loc[show_table] for k, v in summaries.items()}))
    
    return (results, summaries)

def analyse_svm_margin(adata, adata_cor, condition_name, layer=None):
    print(layer)
    if layer is not None:
        adata.layers['default_X'] = adata.X
        adata.X = adata.layers[layer]
        adata_cor.layers['default_X'] = adata_cor.X
        adata_cor.X = adata_cor.layers[layer]

    def get_svm_margin(adata, condition_name, size_factor = 1):
        predictors = adata.X * size_factor
        result = adata.obs[condition_name]
        clf = LinearSVC(random_state=0, dual=False)
        clf.fit(predictors, result)
        margin_df = pd.DataFrame({'condition': clf.classes_, 'margin': 1 / np.sqrt(np.sum(clf.coef_**2, axis=1))})

        #print(margin_df)
        return margin_df

    size_factor = np.sum(adata.X) / np.sum(adata_cor.X)

    df = pd.merge(get_svm_margin(adata, condition_name),
                  get_svm_margin(adata_cor, condition_name, size_factor = size_factor),
                  on='condition', suffixes=['_uncorrected', '_ISM_corrected'])
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.barplot(df.melt(id_vars='condition', var_name='correction', value_name='margin'),
                x='condition', y='margin', hue='correction').set_title('Comparison of SVM margins for layer %s'%layer)



    if layer is not None:
        adata.X = adata.layers['default_X']
        adata_cor.X = adata_cor.layers['default_X']