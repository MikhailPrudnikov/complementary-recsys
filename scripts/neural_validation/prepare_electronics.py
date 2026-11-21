"""Prepare a temporal novel-pair benchmark with train-time product attributes."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import hashlib,json,math
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw", type=Path, required=True, help="Electronics CSV or its single-file ZIP")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--item-id", action="store_true")
args = parser.parse_args()
source, out = args.raw, args.output
out.mkdir(parents=True, exist_ok=False)
e=pd.read_csv(source,dtype='string').drop_duplicates().dropna(subset=['event_time','user_id','product_id','user_session']).copy()
e['time']=pd.to_datetime(e.event_time,utc=True)
times=e.time.sort_values().reset_index(drop=True)
c1,c2=[times.iloc[int(len(times)*v)] for v in [.8,.9]]
e['split']=(e.time>=c1).astype(int)+(e.time>=c2).astype(int)
g=e.groupby(['user_id','user_session']).split
e=e[g.transform('min')==g.transform('max')].copy()
tr=e[e.split==0]
items=tr.sort_values('time').drop_duplicates('product_id',keep='last').dropna(subset='category_id').copy()
count=items.category_id.value_counts()
items=items[items.category_id.isin(count[count>=20].index)].sort_values('product_id').reset_index(drop=True)
ids=dict(zip(items.product_id,items.index))
cat,catnames=pd.factorize(items.category_id,sort=True)
vocab={'PAD':0}
def tokens(row):
    price=float(row.price) if pd.notna(row.price) else 0
    terms=['brand='+str(row.brand),'pricebin='+str(math.floor(8*math.log1p(max(0,price)))),'path='+str(row.category_code)]
    return [vocab.setdefault(t,len(vocab)) for t in terms]
tok=np.array([tokens(r) for r in items.itertuples()],dtype=np.int64)
rawcounts={}; seen=set();arr={'tokens':tok,'categories':cat.astype('int64'),'holdout':np.empty((0,2),dtype='int64')}
for split,name in enumerate(['train','validation','test']):
    for event,relation in [('purchase','buy'),('view','view')]:
        if split and relation=='view': continue
        pairs=set()
        for values in e[(e.split==split)&(e.event_type==event)].groupby(['user_id','user_session']).product_id.unique():
            if len(values)>50 and relation=='view': continue
            ix=sorted(ids[v] for v in values if v in ids)
            for a,b in combinations(ix,2):
                if relation=='view' or cat[a]!=cat[b]:pairs.add((a,b))
        rawcounts[name+'_'+relation]=len(pairs)
        if relation=='buy':
            if split: pairs-=seen
            seen.update(pairs)
        arr[name+'_'+relation]=np.asarray(sorted(pairs|{(b,a) for a,b in pairs}),dtype='int64').reshape(-1,2)
if args.item_id:
    warm = np.unique(np.concatenate([arr["train_buy"].ravel(), arr["train_view"].ravel()]))
    item_tokens = np.full(len(items), len(vocab), dtype=np.int64)
    item_tokens[warm] = len(vocab) + 1 + np.arange(len(warm))
    arr["tokens"] = np.column_stack([tok, item_tokens])
np.savez_compressed(out/'benchmark.npz',**arr)
items=items.rename(columns={'product_id':'item_id','category_id':'category'})
items['title']=items.brand.fillna('?')+' / '+items.category_code.fillna(items.category)+' / price '+items.price.fillna('?')
items[['item_id','category','title','brand','price','category_code']].to_csv(out/'items.csv.gz',index=False)
manifest=dict(dataset='REES46 Electronics',source='mkechinov/ecommerce-events-history-in-electronics-store',raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cutoffs=[str(c1),str(c2)],policy='Full known-item catalog, >=20 products per train category. Latest train attributes. Bidirectional within-session pairs. Novel cross-category buys. Evaluation positives exclude earlier buy periods. Train sampler has no future-pair exclusion.',n_items=len(items),n_categories=len(catnames),vocab_size=len(vocab),category_names=list(catnames),before_novel_filter_unordered=rawcounts,counts={k:len(v) for k,v in arr.items() if k not in ['categories','tokens']},distinct_feature_profiles=len(set(zip(cat,map(tuple,tok)))),benchmark_sha256=hashlib.sha256((out/'benchmark.npz').read_bytes()).hexdigest())
assert all(len(arr[k]) for k in ['train_view','train_buy','validation_buy','test_buy'])
if args.item_id:
    manifest["vocab_size"] = int(arr["tokens"].max()) + 1
    manifest["dataset"] += " with identity"
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print({k:v for k,v in manifest.items() if k!='category_names'})
