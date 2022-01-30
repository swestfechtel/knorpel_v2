from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scikit_posthocs as sp


result_dir = 'C:/Users/simon/Desktop/Ergebnisse'

fn01 = pd.read_csv(f'{result_dir}/function_normals_01.csv')
fn234 = pd.read_csv(f'{result_dir}/function_normals_234.csv')

fv01 = pd.read_csv(f'{result_dir}/function_values_01.csv')
fv234 = pd.read_csv(f'{result_dir}/function_values_234.csv')

m01 = pd.read_csv(f'{result_dir}/mesh_01.csv')
m234 = pd.read_csv(f'{result_dir}/mesh_234.csv')

s01 = pd.read_csv(f'{result_dir}/sphere_01.csv')
s234 = pd.read_csv(f'{result_dir}/sphere_234.csv')

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

fn01 = fn01[(stats.zscore(fn01) < 5).all(axis=1)]
fv01 = fv01[(stats.zscore(fv01) < 5).all(axis=1)]
m01 = m01[(stats.zscore(m01) < 5).all(axis=1)]
s01 = s01[(stats.zscore(s01) < 5).all(axis=1)]

fn234 = fn234[(stats.zscore(fn234) < 5).all(axis=1)]
fv234 = fv234[(stats.zscore(fv234) < 5).all(axis=1)]
m234 = m234[(stats.zscore(m234) < 5).all(axis=1)]
s234 = s234[(stats.zscore(s234) < 5).all(axis=1)]

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

fv01_lf_mean = fv01[lf_cols].apply(np.mean, axis=1)
fv234_lf_mean = fv234[lf_cols].apply(np.mean, axis=1)

fv01_mf_mean = fv01[mf_cols].apply(np.mean, axis=1)
fv234_mf_mean = fv234[mf_cols].apply(np.mean, axis=1)

fv01_lt_mean = fv01[lt_cols].apply(np.mean, axis=1)
fv234_lt_mean = fv234[lt_cols].apply(np.mean, axis=1)

fv01_mt_mean = fv01[mt_cols].apply(np.mean, axis=1)
fv234_mt_mean = fv234[mt_cols].apply(np.mean, axis=1)

m01_lf_mean = m01[lf_cols].apply(np.mean, axis=1)
m234_lf_mean = m234[lf_cols].apply(np.mean, axis=1)

m01_mf_mean = m01[mf_cols].apply(np.mean, axis=1)
m234_mf_mean = m234[mf_cols].apply(np.mean, axis=1)

m01_lt_mean = m01[lt_cols].apply(np.mean, axis=1)
m234_lt_mean = m234[lt_cols].apply(np.mean, axis=1)

m01_mt_mean = m01[mt_cols].apply(np.mean, axis=1)
m234_mt_mean = m234[mt_cols].apply(np.mean, axis=1)

s01_lf_mean = s01[lf_cols].apply(np.mean, axis=1)
s234_lf_mean = s234[lf_cols].apply(np.mean, axis=1)

s01_mf_mean = s01[mf_cols].apply(np.mean, axis=1)
s234_mf_mean = s234[mf_cols].apply(np.mean, axis=1)

s01_lt_mean = s01[lt_cols].apply(np.mean, axis=1)
s234_lt_mean = s234[lt_cols].apply(np.mean, axis=1)

s01_mt_mean = s01[mt_cols].apply(np.mean, axis=1)
s234_mt_mean = s234[mt_cols].apply(np.mean, axis=1)

# ------------------------------------- Plotting ---------------------------------------------------

xticklabels = ['Mesh', '2D function normals', '2D function values', 'Sphere ray tracing']

fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot([m_all_regions_mean, fn_all_regions_mean, fv_all_regions_mean, s_all_regions_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Femur/Tibia combined mean thickness')
fig.savefig(f'plots/all_regions.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot([m_f_mean, fn_f_mean, fv_f_mean, s_f_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Femur mean thickness')
fig.savefig(f'plots/femur_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot([m_t_mean, fn_t_mean, fv_t_mean, s_t_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Tibia mean thickness')
fig.savefig(f'plots/tibia_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot([m_lf_mean, fn_lf_mean, fv_lf_mean, s_lf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Femur mean thickness')
fig.savefig(f'plots/lateral_femur_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot([m_mf_mean, fn_mf_mean, fv_mf_mean, s_mf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Femur mean thickness')
fig.savefig(f'plots/medial_femur_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot([m_lt_mean, fn_lt_mean, fv_lt_mean, s_lt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Tibia mean thickness')
fig.savefig(f'plots/lateral_tibia_mean.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot([m_mt_mean, fn_mt_mean, fv_mt_mean, s_mt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Tibia mean thickness')
fig.savefig(f'plots/lateral_tibia_mean.png')
plt.close(fig)

for subregion in all_cols:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.boxplot([m[subregion], fn[subregion], fv[subregion], s[subregion]])
    ax.set_xticklabels(xticklabels)
    ax.vlines(x=1.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
    ax.set_title(f'{subregion} mean thickness')
    fig.savefig(f'plots/{subregion}_mean.png')
    plt.close(fig)

xticklabels = ['Mesh 0,1', '2D function normals 0,1', '2D function values 0,1', 'Sphere ray tracing 0,1', 'Mesh 2,3,4', '2D function normals 2,3,4', '2D function values 2,3,4', 'Sphere ray tracing 2,3,4']

fig, ax = plt.subplots(figsize=(20, 10))
ax.boxplot([m01_lf_mean, fn01_lf_mean, fv01_lf_mean, s01_lf_mean, m234_lf_mean, fn234_lf_mean, fv234_lf_mean, s234_lf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=4.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Femur rOA grades 0,1 vs rOA grades 2,3,4')
fig.savefig(f'plots/lateral_femur_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(20, 10))
ax.boxplot([m01_mf_mean, fn01_mf_mean, fv01_mf_mean, s01_mf_mean, m234_mf_mean, fn234_mf_mean, fv234_mf_mean, s234_mf_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=4.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Femur rOA grades 0,1 vs rOA grades 2,3,4')
fig.savefig(f'plots/medial_femur_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(20, 10))
ax.boxplot([m01_lt_mean, fn01_lt_mean, fv01_lt_mean, s01_lt_mean, m234_lt_mean, fn234_lt_mean, fv234_lt_mean, s234_lt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=4.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Lateral Tibia rOA grades 0,1 vs rOA grades 2,3,4')
fig.savefig(f'plots/lateral_tibia_roa.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(20, 10))
ax.boxplot([m01_mt_mean, fn01_mt_mean, fv01_mt_mean, s01_mt_mean, m234_mt_mean, fn234_mt_mean, fv234_mt_mean, s234_mt_mean])
ax.set_xticklabels(xticklabels)
ax.vlines(x=4.5, ymin=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .1, ymax=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .9, linestyles='dashed', colors=['dimgray'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f mm'))
ax.set_title('Medial Tibia rOA grades 0,1 vs rOA grades 2,3,4')
fig.savefig(f'plots/medial_tibia_roa.png')
plt.close(fig)