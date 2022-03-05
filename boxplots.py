from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scikit_posthocs as sp


result_dir = 'C:/Users/simon/Desktop/Ergebnisse'

fn0 = pd.read_csv(f'{result_dir}/fn_roa0.csv')
fn4 = pd.read_csv(f'{result_dir}/fn_roa4.csv')
fn01 = pd.read_csv(f'{result_dir}/function_normals_01_new.csv')
fn234 = pd.read_csv(f'{result_dir}/function_normals_234_new.csv')

fv01 = pd.read_csv(f'{result_dir}/function_values_01.csv')
fv234 = pd.read_csv(f'{result_dir}/function_values_234.csv')
fv0 = pd.read_csv(f'{result_dir}/fv_roa0.csv')
fv4 = pd.read_csv(f'{result_dir}/fv_roa4.csv')

m01 = pd.read_csv(f'{result_dir}/mesh_01.csv')
m234 = pd.read_csv(f'{result_dir}/mesh_234.csv')
m0 = pd.read_csv(f'{result_dir}/mesh_roa0.csv')
m4 = pd.read_csv(f'{result_dir}/mesh_roa4.csv')

s01 = pd.read_csv(f'{result_dir}/sphere_01.csv')
s234 = pd.read_csv(f'{result_dir}/sphere_234.csv')
s0 = pd.read_csv(f'{result_dir}/sphere_roa0.csv')
s4 = pd.read_csv(f'{result_dir}/sphere_roa4.csv')

fn = pd.concat([fn01, fn234])
fv = pd.concat([fv01, fv234])
m = pd.concat([m01, m234])
s = pd.concat([s01, s234])

fn01.set_index('dir', inplace=True)
fv01.set_index('dir', inplace=True)
m01.set_index('dir', inplace=True)
s01.set_index('dir', inplace=True)

fn234.set_index('dir', inplace=True)
fv234.set_index('dir', inplace=True)
m234.set_index('dir', inplace=True)
s234.set_index('dir', inplace=True)

fn0.set_index('dir', inplace=True)
fv0.set_index('dir', inplace=True)
m0.set_index('dir', inplace=True)
s0.set_index('dir', inplace=True)

fn4.set_index('dir', inplace=True)
fv4.set_index('dir', inplace=True)
m4.set_index('dir', inplace=True)
s4.set_index('dir', inplace=True)

fn.set_index('dir', inplace=True)
fv.set_index('dir', inplace=True)
m.set_index('dir', inplace=True)
s.set_index('dir', inplace=True)

fn01 = fn01[fn01.isna().any(axis=1) == False]
fv01 = fv01[fv01.isna().any(axis=1) == False]
m01 = m01[m01.isna().any(axis=1) == False]
s01 = s01[s01.isna().any(axis=1) == False]

fn234 = fn234[fn234.isna().any(axis=1) == False]
fv234 = fv234[fv234.isna().any(axis=1) == False]
m234 = m234[m234.isna().any(axis=1) == False]
s234 = s234[s234.isna().any(axis=1) == False]

fn0 = fn0[fn0.isna().any(axis=1) == False]
fv0 = fv0[fv0.isna().any(axis=1) == False]
m0 = m0[m0.isna().any(axis=1) == False]
s0 = s0[s0.isna().any(axis=1) == False]

fn4 = fn4[fn4.isna().any(axis=1) == False]
fv4 = fv4[fv4.isna().any(axis=1) == False]
m4 = m4[m4.isna().any(axis=1) == False]
s4 = s4[s4.isna().any(axis=1) == False]

fn01 = fn01[(stats.zscore(fn01) < 5).all(axis=1)]
fv01 = fv01[(stats.zscore(fv01) < 5).all(axis=1)]
m01 = m01[(stats.zscore(m01) < 5).all(axis=1)]
s01 = s01[(stats.zscore(s01) < 5).all(axis=1)]

fn234 = fn234[(stats.zscore(fn234) < 5).all(axis=1)]
fv234 = fv234[(stats.zscore(fv234) < 5).all(axis=1)]
m234 = m234[(stats.zscore(m234) < 5).all(axis=1)]
s234 = s234[(stats.zscore(s234) < 5).all(axis=1)]

fn0 = fn0[(stats.zscore(fn0) < 5).all(axis=1)]
fv0 = fv0[(stats.zscore(fv0) < 5).all(axis=1)]
m0 = m0[(stats.zscore(m0) < 5).all(axis=1)]
s0 = s0[(stats.zscore(s0) < 5).all(axis=1)]

fn4 = fn4[(stats.zscore(fn4) < 5).all(axis=1)]
fv4 = fv4[(stats.zscore(fv4) < 5).all(axis=1)]
m4 = m4[(stats.zscore(m4) < 5).all(axis=1)]
s4 = s4[(stats.zscore(s4) < 5).all(axis=1)]

intersection = fn01.index.intersection(fv01.index.intersection(m01.index.intersection(s01.index)))
fn01 = fn01.loc[intersection]
fv01 = fv01.loc[intersection]
m01 = m01.loc[intersection]
s01 = s01.loc[intersection]

intersection = fn234.index.intersection(fv234.index.intersection(m234.index.intersection(s234.index)))
fn234 = fn234.loc[intersection]
fv234 = fv234.loc[intersection]
m234 = m234.loc[intersection]
s234 = s234.loc[intersection]

intersection = fn0.index.intersection(fv0.index.intersection(m0.index.intersection(s0.index)))
fn0 = fn0.loc[intersection]
fv0 = fv0.loc[intersection]
m0 = m0.loc[intersection]
s0 = s0.loc[intersection]

intersection = fn4.index.intersection(fv4.index.intersection(m4.index.intersection(s4.index)))
fn4 = fn4.loc[intersection]
fv4 = fv4.loc[intersection]
m4 = m4.loc[intersection]
s4 = s4.loc[intersection]

fn = fn[fn.isna().any(axis=1) == False]
fv = fv[fv.isna().any(axis=1) == False]
m = m[m.isna().any(axis=1) == False]
s = s[s.isna().any(axis=1) == False]

numrows = fn.shape[0]
fn = fn[(stats.zscore(fn) < 5).all(axis=1)]
print(f'Filtered out {numrows - fn.shape[0]} instances.')

numrows = fv.shape[0]
fv = fv[(stats.zscore(fv) < 5).all(axis=1)]
print(f'Filtered out {numrows - fv.shape[0]} instances.')

numrows = m.shape[0]
m = m[(stats.zscore(m) < 5).all(axis=1)]
print(f'Filtered out {numrows - m.shape[0]} instances.')

numrows = s.shape[0]
s = s[(stats.zscore(s) < 5).all(axis=1)]
print(f'Filtered out {numrows - s.shape[0]} instances.')

intersection = fn.index.intersection(fv.index.intersection(m.index.intersection(s.index)))

fn = fn.loc[intersection]
fv = fv.loc[intersection]
m = m.loc[intersection]
s = s.loc[intersection]

assert fn.shape == fv.shape == m.shape == s.shape

all_cols = ['ecLF', 'ccLF', 'icLF', 'ecMF', 'ccMF', 'icMF', 'aLF', 'aMF', 'pLF', 'pMF', 
           'cLT', 'iLT', 'eLT', 'aLT', 'pLT', 'cMT', 'iMT', 'eMT', 'aMT', 'pMT']

f_cols = ['ecLF', 'ccLF', 'icLF', 'ecMF', 'ccMF', 'icMF', 'aLF', 'aMF', 'pLF', 'pMF']
t_cols = ['cLT', 'iLT', 'eLT', 'aLT', 'pLT', 'cMT', 'iMT', 'eMT', 'aMT', 'pMT']

lf_cols = ['ecLF', 'ccLF', 'icLF', 'aLF', 'pLF']
mf_cols = ['ecMF', 'ccMF', 'icMF', 'aMF', 'pMF']
lt_cols = ['cLT', 'iLT', 'eLT', 'aLT', 'pLT']
mt_cols = ['cMT', 'iMT', 'eMT', 'aMT', 'pMT']

fn_all_regions_mean = fn[all_cols].apply(np.mean, axis=1)
fv_all_regions_mean = fv[all_cols].apply(np.mean, axis=1)
m_all_regions_mean = m[all_cols].apply(np.mean, axis=1)
s_all_regions_mean = s[all_cols].apply(np.mean, axis=1)

fn_f_mean = fn[f_cols].apply(np.mean, axis=1)
fn_t_mean = fn[t_cols].apply(np.mean, axis=1)

fv_f_mean = fv[f_cols].apply(np.mean, axis=1)
fv_t_mean = fv[t_cols].apply(np.mean, axis=1)

m_f_mean = m[f_cols].apply(np.mean, axis=1)
m_t_mean = m[t_cols].apply(np.mean, axis=1)

s_f_mean = s[f_cols].apply(np.mean, axis=1)
s_t_mean = s[t_cols].apply(np.mean, axis=1)

fn_lf_mean = fn[lf_cols].apply(np.mean, axis=1)
fn_mf_mean = fn[mf_cols].apply(np.mean, axis=1)
fn_lt_mean = fn[lt_cols].apply(np.mean, axis=1)
fn_mt_mean = fn[mt_cols].apply(np.mean, axis=1)

fv_lf_mean = fv[lf_cols].apply(np.mean, axis=1)
fv_mf_mean = fv[mf_cols].apply(np.mean, axis=1)
fv_lt_mean = fv[lt_cols].apply(np.mean, axis=1)
fv_mt_mean = fv[mt_cols].apply(np.mean, axis=1)

m_lf_mean = m[lf_cols].apply(np.mean, axis=1)
m_mf_mean = m[mf_cols].apply(np.mean, axis=1)
m_lt_mean = m[lt_cols].apply(np.mean, axis=1)
m_mt_mean = m[mt_cols].apply(np.mean, axis=1)

s_lf_mean = s[lf_cols].apply(np.mean, axis=1)
s_mf_mean = s[mf_cols].apply(np.mean, axis=1)
s_lt_mean = s[lt_cols].apply(np.mean, axis=1)
s_mt_mean = s[mt_cols].apply(np.mean, axis=1)

fn_all_regions = fn[all_cols]
fv_all_regions = fv[all_cols]
m_all_regions = m[all_cols]
s_all_regions = s[all_cols]

fn01_lf_mean = fn01[lf_cols].apply(np.mean, axis=1)
fn234_lf_mean = fn234[lf_cols].apply(np.mean, axis=1)

fn01_mf_mean = fn01[mf_cols].apply(np.mean, axis=1)
fn234_mf_mean = fn234[mf_cols].apply(np.mean, axis=1)

fn01_lt_mean = fn01[lt_cols].apply(np.mean, axis=1)
fn234_lt_mean = fn234[lt_cols].apply(np.mean, axis=1)

fn01_mt_mean = fn01[mt_cols].apply(np.mean, axis=1)
fn234_mt_mean = fn234[mt_cols].apply(np.mean, axis=1)

fn01_clf_mean = fn01[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
fn234_clf_mean = fn234[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

fn01_cmf_mean = fn01[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
fn234_cmf_mean = fn234[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

fv01_lf_mean = fv01[lf_cols].apply(np.mean, axis=1)
fv234_lf_mean = fv234[lf_cols].apply(np.mean, axis=1)

fv01_mf_mean = fv01[mf_cols].apply(np.mean, axis=1)
fv234_mf_mean = fv234[mf_cols].apply(np.mean, axis=1)

fv01_lt_mean = fv01[lt_cols].apply(np.mean, axis=1)
fv234_lt_mean = fv234[lt_cols].apply(np.mean, axis=1)

fv01_mt_mean = fv01[mt_cols].apply(np.mean, axis=1)
fv234_mt_mean = fv234[mt_cols].apply(np.mean, axis=1)

fv01_clf_mean = fv01[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
fv234_clf_mean = fv234[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

fv01_cmf_mean = fv01[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
fv234_cmf_mean = fv234[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

m01_lf_mean = m01[lf_cols].apply(np.mean, axis=1)
m234_lf_mean = m234[lf_cols].apply(np.mean, axis=1)

m01_mf_mean = m01[mf_cols].apply(np.mean, axis=1)
m234_mf_mean = m234[mf_cols].apply(np.mean, axis=1)

m01_lt_mean = m01[lt_cols].apply(np.mean, axis=1)
m234_lt_mean = m234[lt_cols].apply(np.mean, axis=1)

m01_mt_mean = m01[mt_cols].apply(np.mean, axis=1)
m234_mt_mean = m234[mt_cols].apply(np.mean, axis=1)

m01_clf_mean = m01[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
m234_clf_mean = m234[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

m01_cmf_mean = m01[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
m234_cmf_mean = m234[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

s01_lf_mean = s01[lf_cols].apply(np.mean, axis=1)
s234_lf_mean = s234[lf_cols].apply(np.mean, axis=1)

s01_mf_mean = s01[mf_cols].apply(np.mean, axis=1)
s234_mf_mean = s234[mf_cols].apply(np.mean, axis=1)

s01_lt_mean = s01[lt_cols].apply(np.mean, axis=1)
s234_lt_mean = s234[lt_cols].apply(np.mean, axis=1)

s01_mt_mean = s01[mt_cols].apply(np.mean, axis=1)
s234_mt_mean = s234[mt_cols].apply(np.mean, axis=1)

s01_clf_mean = s01[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
s234_clf_mean = s234[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

s01_cmf_mean = s01[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
s234_cmf_mean = s234[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

fn0_lf_mean = fn0[lf_cols].apply(np.mean, axis=1)
fn4_lf_mean = fn4[lf_cols].apply(np.mean, axis=1)

fn0_mf_mean = fn0[mf_cols].apply(np.mean, axis=1)
fn4_mf_mean = fn4[mf_cols].apply(np.mean, axis=1)

fn0_lt_mean = fn0[lt_cols].apply(np.mean, axis=1)
fn4_lt_mean = fn4[lt_cols].apply(np.mean, axis=1)

fn0_mt_mean = fn0[mt_cols].apply(np.mean, axis=1)
fn4_mt_mean = fn4[mt_cols].apply(np.mean, axis=1)

fn0_clf_mean = fn0[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
fn4_clf_mean = fn4[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

fn0_cmf_mean = fn0[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
fn4_cmf_mean = fn4[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

fv0_lf_mean = fv0[lf_cols].apply(np.mean, axis=1)
fv4_lf_mean = fv4[lf_cols].apply(np.mean, axis=1)

fv0_mf_mean = fv0[mf_cols].apply(np.mean, axis=1)
fv4_mf_mean = fv4[mf_cols].apply(np.mean, axis=1)

fv0_lt_mean = fv0[lt_cols].apply(np.mean, axis=1)
fv4_lt_mean = fv4[lt_cols].apply(np.mean, axis=1)

fv0_mt_mean = fv0[mt_cols].apply(np.mean, axis=1)
fv4_mt_mean = fv4[mt_cols].apply(np.mean, axis=1)

fv0_clf_mean = fv0[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
fv4_clf_mean = fv4[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

fv0_cmf_mean = fv0[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
fv4_cmf_mean = fv4[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

m0_lf_mean = m0[lf_cols].apply(np.mean, axis=1)
m4_lf_mean = m4[lf_cols].apply(np.mean, axis=1)

m0_mf_mean = m0[mf_cols].apply(np.mean, axis=1)
m4_mf_mean = m4[mf_cols].apply(np.mean, axis=1)

m0_lt_mean = m0[lt_cols].apply(np.mean, axis=1)
m4_lt_mean = m4[lt_cols].apply(np.mean, axis=1)

m0_mt_mean = m0[mt_cols].apply(np.mean, axis=1)
m4_mt_mean = m4[mt_cols].apply(np.mean, axis=1)

m0_clf_mean = m0[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
m4_clf_mean = m4[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

m0_cmf_mean = m0[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
m4_cmf_mean = m4[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

s0_lf_mean = s0[lf_cols].apply(np.mean, axis=1)
s4_lf_mean = s4[lf_cols].apply(np.mean, axis=1)

s0_mf_mean = s0[mf_cols].apply(np.mean, axis=1)
s4_mf_mean = s4[mf_cols].apply(np.mean, axis=1)

s0_lt_mean = s0[lt_cols].apply(np.mean, axis=1)
s4_lt_mean = s4[lt_cols].apply(np.mean, axis=1)

s0_mt_mean = s0[mt_cols].apply(np.mean, axis=1)
s4_mt_mean = s4[mt_cols].apply(np.mean, axis=1)

s0_clf_mean = s0[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)
s4_clf_mean = s4[['ccLF', 'icLF', 'ecLF']].apply(np.mean, axis=1)

s0_cmf_mean = s0[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)
s4_cmf_mean = s4[['ccMF', 'icMF', 'ecMF']].apply(np.mean, axis=1)

# ------------------------------------- Plotting ---------------------------------------------------

xticklabels = ['3D-MN', '3D-RT', '2D-FN', '2D-FS']
region_figsize = (10, 10)
roa_figsize = (20, 10)

fig, ax = plt.subplots(figsize=region_figsize)
ax.boxplot([m_all_regions_mean, s_all_regions_mean, fn_all_regions_mean, fv_all_regions_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Femur/Tibia combined')
ax.set_xlabel('Method')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/all_regions.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=region_figsize)
ax.boxplot([m_f_mean, s_f_mean, fn_f_mean, fv_f_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Femur')
ax.set_xlabel('Method')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/femur_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=region_figsize)
ax.boxplot([m_t_mean, s_t_mean, fn_t_mean, fv_t_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Tibia')
ax.set_xlabel('Method')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/tibia_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=region_figsize)
ax.boxplot([m_lf_mean, s_lf_mean, fn_lf_mean, fv_lf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Femur')
ax.set_xlabel('Method')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/lateral_femur_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=region_figsize)
ax.boxplot([m_mf_mean, s_mf_mean, fn_mf_mean, fv_mf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Femur')
ax.set_xlabel('Method')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/medial_femur_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=region_figsize)
ax.boxplot([m_lt_mean, s_lt_mean, fn_lt_mean, fv_lt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Tibia')
ax.set_xlabel('Method')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/lateral_tibia_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=region_figsize)
ax.boxplot([m_mt_mean, s_mt_mean, fn_mt_mean, fv_mt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Tibia')
ax.set_xlabel('Method')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/lateral_tibia_mean.png')
plt.close(fig)

for subregion in all_cols:
    fig, ax = plt.subplots(figsize=region_figsize)
    ax.boxplot([m[subregion], s[subregion], fn[subregion], fv[subregion]])
    ax.set_xticklabels(xticklabels)
    ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
    ax.set_title(f'{subregion}')
    ax.set_xlabel('Method')
    ax.set_ylabel('Mean thickness [mm]')
    fig.savefig(f'plots/{subregion}_mean.png')
    plt.close(fig)

xticklabels = ['3D-MN 0,1', '3D-MN 2,3,4', '3D-RT 0,1', '3D-RT 2,3,4', '2D-FN 0,1', '2D-FN 2,3,4', '2D-FS 0,1', '2D-FS 2,3,4']

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_lf_mean, fn01_lf_mean, fv01_lf_mean, s01_lf_mean, m234_lf_mean, fn234_lf_mean, fv234_lf_mean, s234_lf_mean])
ax.boxplot([m01_lf_mean, m234_lf_mean, s01_lf_mean, s234_lf_mean, fn01_lf_mean, fn234_lf_mean, fv01_lf_mean, fv234_lf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Femur rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/lateral_femur_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mf_mean, fn01_mf_mean, fv01_mf_mean, s01_mf_mean, m234_mf_mean, fn234_mf_mean, fv234_mf_mean, s234_mf_mean])
ax.boxplot([m01_mf_mean, m234_mf_mean, s01_mf_mean, s234_mf_mean, fn01_mf_mean, fn234_mf_mean, fv01_mf_mean, fv234_mf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Femur rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/medial_femur_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_lt_mean, fn01_lt_mean, fv01_lt_mean, s01_lt_mean, m234_lt_mean, fn234_lt_mean, fv234_lt_mean, s234_lt_mean])
ax.boxplot([m01_lt_mean, m234_lt_mean, s01_lt_mean, s234_lt_mean, fn01_lt_mean, fn234_lt_mean, fv01_lt_mean, fv234_lt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Tibia rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/lateral_tibia_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m01_mt_mean, m234_mt_mean, s01_mt_mean, s234_mt_mean, fn01_mt_mean, fn234_mt_mean, fv01_mt_mean, fv234_mt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Tibia rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/medial_tibia_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m01['cLT'], m234['cLT'], s01['cLT'], s234['cLT'], fn01['cLT'], fn234['cLT'], fv01['cLT'], fv234['cLT']])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cLT rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/clt_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m01['cMT'], m234['cMT'], s01['cMT'], s234['cMT'], fn01['cMT'], fn234['cMT'], fv01['cMT'], fv234['cMT']])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cMT rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/cmt_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m01_clf_mean, m234_clf_mean, s01_clf_mean, s234_clf_mean, fn01_clf_mean, fn234_clf_mean, fv01_clf_mean, fv234_clf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cLF rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/clf_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m01_cmf_mean, m234_cmf_mean, s01_cmf_mean, s234_cmf_mean, fn01_cmf_mean, fn234_cmf_mean, fv01_cmf_mean, fv234_cmf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cMF rOA grades 0,1 vs rOA grades 2,3,4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/cmf_roa.png')
plt.close(fig)

# --

xticklabels = ['3D-MN 0', '3D-MN 4', '3D-RT 0', '3D-RT 4', '2D-FN 0', '2D-FN 4', '2D-FS 0', '2D-FS 4']

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_lf_mean, fn01_lf_mean, fv01_lf_mean, s01_lf_mean, m234_lf_mean, fn234_lf_mean, fv234_lf_mean, s234_lf_mean])
ax.boxplot([m0_lf_mean, m4_lf_mean, s0_lf_mean, s4_lf_mean, fn0_lf_mean, fn4_lf_mean, fv0_lf_mean, fv4_lf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Femur rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/lateral_femur_roa_0v4.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mf_mean, fn01_mf_mean, fv01_mf_mean, s01_mf_mean, m234_mf_mean, fn234_mf_mean, fv234_mf_mean, s234_mf_mean])
ax.boxplot([m0_mf_mean, m4_mf_mean, s0_mf_mean, s4_mf_mean, fn0_mf_mean, fn4_mf_mean, fv0_mf_mean, fv4_mf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Femur rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/medial_femur_roa_0v4.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_lt_mean, fn01_lt_mean, fv01_lt_mean, s01_lt_mean, m234_lt_mean, fn234_lt_mean, fv234_lt_mean, s234_lt_mean])
ax.boxplot([m0_lt_mean, m4_lt_mean, s0_lt_mean, s4_lt_mean, fn0_lt_mean, fn4_lt_mean, fv0_lt_mean, fv4_lt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Tibia rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/lateral_tibia_roa_0v4.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m0_mt_mean, m4_mt_mean, s0_mt_mean, s4_mt_mean, fn0_mt_mean, fn4_mt_mean, fv0_mt_mean, fv4_mt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Tibia rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/medial_tibia_roa_0v4.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m0['cLT'], m4['cLT'], s0['cLT'], s4['cLT'], fn0['cLT'], fn4['cLT'], fv0['cLT'], fv4['cLT']])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cLT rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/clt_roa_0v4.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m0['cMT'], m4['cMT'], s0['cMT'], s4['cMT'], fn0['cMT'], fn4['cMT'], fv0['cMT'], fv4['cMT']])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cMT rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/cmt_roa_0v4.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m0_clf_mean, m4_clf_mean, s0_clf_mean, s4_clf_mean, fn0_clf_mean, fn4_clf_mean, fv0_clf_mean, fv4_clf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cLF rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/clf_roa_0v4.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=roa_figsize)
# ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.boxplot([m0_cmf_mean, m4_cmf_mean, s0_cmf_mean, s4_cmf_mean, fn0_cmf_mean, fn4_cmf_mean, fv0_cmf_mean, fv4_cmf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=2.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('cMF rOA grades 0 vs rOA grades 4')
ax.set_xlabel('Method/rOA grade')
ax.set_ylabel('Mean thickness [mm]')
fig.savefig(f'plots/cmf_roa_0v4.png')
plt.close(fig)