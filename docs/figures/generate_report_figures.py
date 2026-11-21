"""Figures from saved dataset_analysis.ipynb outputs and split CSVs; no model runs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
names = ['Amazon\nMusical Instruments', 'Retailrocket', 'REES46\nCosmetics', 'REES46\nElectronics']
buy = np.array([82893, 7052, 7779980, 4053])
view = np.array([227621, 1034911, 13113113, 260116])
items = np.array([112135, 235061, 54571, 53453])
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, ax = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
x=np.arange(4)
ax[0].bar(x-.18,buy,.36,label='Покупки',color='#315a85')
ax[0].bar(x+.18,view,.36,label='Просмотры',color='#d58a36')
ax[0].set_yscale('log'); ax[0].set_xticks(x,names); ax[0].set_ylabel('Уникальные неупорядоченные связи (log)')
ax[0].set_title('а. Объём двух типов связей');ax[0].legend()
density=100*buy/(items*(items-1)/2)
ax[1].bar(x,density,color='#315a85');ax[1].set_yscale('log');ax[1].set_xticks(x,names)
ax[1].set_title('б. Плотность графа покупок');ax[1].set_ylabel('Доля всех возможных пар, % (log)')
ax[1].set_ylim(1e-5,2)
for i,v in enumerate(density):ax[1].text(i,v*1.25,f'{v:.6f}%',ha='center',fontsize=9)
for a in ax:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
fig.savefig(OUT/'data_audit.png',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained')
total=np.array([747,1015086,740]); known=np.array([120,549360,73]); share=known/total*100
bars=ax.barh(['Retailrocket','REES46 Cosmetics','REES46 Electronics'],share,color='#315a85')
ax.set_xlim(0,100);ax.set_xlabel('Доля тестовых buy-наблюдений с known_user=True, %')
ax.set_title('Покрытие тестовых пар известными пользователями')
for b,k,n,p in zip(bars,known,total,share):ax.text(p+1,b.get_y()+b.get_height()/2,f'{p:.1f}% ({k:,} / {n:,})'.replace(',',' '),va='center')
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.savefig(OUT/'known_users.png',dpi=180)
