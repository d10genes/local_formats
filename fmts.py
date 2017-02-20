
# coding: utf-8

# After years of CSV file format domination in the python world, I'm finally seeing more compelling options in the form of local column-oriented data storage. Some give advantages in read speed, file size or both. 
# 
# While I've had good results in these formats for floats and ints, a bottleneck I still have is storing columns with strings. The file format landscape is improving fast, so I won't be surprised if the speed results of the benchmarks here are obsolete before they're published, but this can at least serve as a performance snapshot.
# 
# [Feather](https://github.com/wesm/feather) is a storage format that aims for speed and interoperability between languages based on the Apache Arrow spec. It shares a creator [Wes McKinney](https://github.com/wesm) with Pandas, and integrates nicely into the python data ecosystem.
# 
# [Fastparquet](https://github.com/dask/fastparquet) is a recent fork of python's implementation of the parquet file format, with an unsurprising focus on speed. Both parquet and [Bcolz](https://github.com/Blosc/bcolz) offer compression as a feature, potentially speeding up IO and shrinking the required size on disk.
# 
# 
# ## String data
# A cursory look at these formats shows an obvious performance increase for numerical data, but I quickly found that performance is much more hit-or-miss for data involving strings. My hope here was to get an empirical look at my options when dealing with string data.
# 
# A difficulty with using strings in Python is that pandas seems to treat them as arbitrary python objects, eliminating any performance gains you can get from numerical datatypes using numpy underneath. A couple of options that you have are to convert the series to a categorical data type, or convert all of the string elements to bytes.
# 
# In this post I compare the performance of these different file formats on string data encoded as strings, categoricals, and bytes. While I sometimes treat categorical encoding as a silver bullet for performance, since it is represented internally as integers, the benefits only outweigh the overhead when there are just a few unique strings in a column that are repeated many times. To compare the performance of the different formats in both scenarios, I test them on a series with mostly unique strings, and another with just a few repeated strings. 
# 
# The benchmarks below compare pandas' CSV, feather, fastparquet and bcolz. While the ultimate file size is important, it is secondary to speed for me, so I also ignored compression options like gzip and brotli.
# 
# I'm only using series with about 200k rows because I don't have all day to wait on the benchmarks to finish. Also note that with the combinatorial explosion of possible options, these benchmarks only represent a comparison of the narrow range of options that I would consider.

# In[ ]:

# import datetime as dt
import imports; imports.reload(imports); from imports import *
import utils; reload(utils); from utils import (
    Timer, get_dtypes, mod_cols, add_nulls, get_obj_type,
    plot_scatter, part, getsize, INFEASIBLE, nan,
    check_args, BEnc, StrEnc, combine_rankings
)

import sys

def fprint(*a, **k):
    print(*a, **k)
    sys.stdout.flush()
    
get_ipython().magic('matplotlib inline')
# ;;


# In[ ]:

def null_type(s: pd.Series) -> (bool, type):
    n = (~(s == s)).sum() > 0
    t1 = s.dtype
    if t1 == object:
        tp = type(s.iloc[0])
        return n, tp
    return n, t1


# # Load and set up df

# ### String columns
# Here's the base dataframe with a column for unique values (`Unq`) and one for repetitive values (`Low_card`).

# In[ ]:

nr.seed(0)
N = 200000
chars = np.array(list(string.ascii_lowercase) + lmap(str, range(10)))
nchars = len(chars)
rix = nr.randint(0, high=nchars, size=(N, 32))
rstrs = [''.join(chars[ix]) for ix in rix]

cols = ['Unq', 'Low_card']
df = DataFrame(dict(Unq=rstrs, Low_card=nr.choice(rstrs[:100], size=N)))[cols]


# In[ ]:

df.apply(lambda x: x.nunique())


# ### Bytes | categorical | strings & Nulls | Nonulls
# This function takes a named Series with string data, and returns a DataFrame whose columns are combinations of the different string data encodings, and some with nulls randomly inserted, since this can have a big effect on parquet's performance.

# In[ ]:

def new_cols(s):
    df = s.to_frame()
    c = df.columns[0]
    df = df.assign(
        Str_nls=lambda x: add_nulls(x[c]),
        Bytes=lambda x: x[c].str.encode('utf-8'),
        Cat=lambda x: tocat(x[c])
    ).assign(
        Bytes_nls=lambda x: add_nulls(x['Bytes']),
        Cat_nls=lambda x: add_nulls(x['Cat']),
    )
    return df.rename(columns={c: 'Str'})

tocat = lambda x: x.astype('category')


# In[ ]:

df_types = new_cols(df.Unq)
DataFrame({'Nulls': df_types.isnull().sum(), 'Dtypes': df_types.dtypes})


# Here's a summary:

# ## Parquet writer/reader
# Here are some helper functions to read and write with fastparquet using different options. Specifying the maximum length of a string in the column with `fixed_text` gives a big speed boost, but comes with the cost of finding this length before saving.

# In[ ]:

def max_len(s, nulls=True):
    if nulls:
        s = s[s == s]
    return s.map(len).max()

def pq_writer(get_lens=False, **kw):
    def write_pq(df, fn):
        if get_lens:
            lns = {c: max_len(df[c], nulls=True) for c in get_dtypes(df, object)}
        else:
            lns = None
        fastparquet.write(fn, df, fixed_text=lns, **kw)
    return write_pq

def pq_reader(categories=None):
    def read_pq(fn):
        f = fastparquet.ParquetFile(fn)
        return f.to_pandas(categories=categories)
    return read_pq


# ## Bcolz
# I was originally going to write off bcolz because of some stunningly bad default performance on the data I initially tried. Simply creating a `carray` from a column takes about 20x as long as feather's writing to disk, for example:

# In[ ]:

get_ipython().magic('time _ = bcolz.carray(df.Unq)')


# In[ ]:

get_ipython().magic("time feather.write_dataframe(df, '/tmp/derp.ftr')")


# But after playing around with every idea I could think of, I eventually found out that saving the underlying array after converting its type yields more reasonable results:

# In[ ]:

get_ipython().magic("time _ = bcolz.carray(df.Unq.values.astype('U'), mode='r')")


# As does saving a list rather than a Series:

# In[ ]:

get_ipython().magic("time _ = bcolz.carray(df.Unq.tolist(), mode='r')")


# But because some of the settings are orders of magnitude slower, though, I had to write a somewhat convoluted `feasible_bcolz` function designate which combinations of settings to avoid.

# In[ ]:

tolist = lambda x: x.tolist()
to_unicode = lambda x: x.values.astype('U')
mk_bcolz_writer = lambda **kw: part(write_bcolz, **kw)

    
@check_args
def write_bcolz(df, fn=None, method: BEnc=None, str_enc: StrEnc=None,
                   null: bool=None):
    if not feasible_bcolz(method=method, str_enc=str_enc, null=null):
        return INFEASIBLE
    if method == BEnc.df:
        ct = bcolz.ctable.fromdataframe(df, rootdir=fn)
    else:
        converter = {BEnc.list: tolist , BEnc.utf8: to_unicode}[method]
        cs = [converter(col) if (col.dtype == 'O')
              else col for _, col in df.iteritems()]
        ct = bcolz.ctable(columns=cs, names=list(df), rootdir=fn)
    return ct


def read_bcolz(fn):
    ct = bcolz.open(fn, mode='r')
    return DataFrame(ct[:])


@check_args
def feasible_bcolz(method: BEnc=None, str_enc: StrEnc=None,
                   null: bool=None, **_):
    "Some bcolz settings are unbearably slow."   
    if method == BEnc.df:
        return (str_enc, null) == (StrEnc.str, False)
    elif method == BEnc.list:
        return (str_enc in {StrEnc.str, StrEnc.byte}) and (not null)
    elif method == BEnc.utf8:
        return str_enc != StrEnc.cat
    raise TypeError("Shouldn't reach here")

# feasible_bcolz(method=BEnc.df, str_enc=StrEnc.str, null=False)


# ### Benchmarkers
# And here are some functions to automate the benchmarking, iterating through different inputs and read/write options. I wrapped them all in a sloppy try/except block in a fit of undisciplined laziness.

# In[ ]:

def bench(fn, df, writer, reader, desc=''):
    twrite = Timer(start=1)
    res = writer(df, fn)
    if res is INFEASIBLE:
        # print('INFEASIBLE')
        return nan, nan, nan
    twrite.end()
    # print('Written with', writer)
    
    tread = Timer(start=1)
    rdf = reader(fn)
    tread.end()

    assert df.shape == rdf.shape, '{} != {}'.format(df.shape, rdf.shape)
    # assert (df.dtypes == rdf.dtypes).all(), '{}\n\n != \n\n{}'.format(df.dtypes, rdf.dtypes)
    
    return twrite.time, tread.time, getsize(fn) / 10**6

def try_bench(*a, **kw):
    known_errors = [
        'with dtype bytes',
        "'NoneType' object has no attribute 'encode'",
        "'bytes' object has no attribute 'encode'",
    ]
    try:
        return bench(*a, **kw)
    except Exception as e:
        print(e)
        print(a[0])
        if any(s in str(e) for s in known_errors):
            na = float('nan')
            return na, na, na
        else:
            raise(e)


# In[ ]:

def todf(x):
    d = DataFrame(x).T
    d.columns = ['Write_time', 'Read_time', 'Mb']
    d.index.name = 'Fmt'
    return d.reset_index(drop=0)

def stack_results(res):
    return pd.concat([
        (df.assign(Enc=type)
           .assign(Null=null))
        for df, type, null in res
    ], ignore_index=True)


# In[ ]:

@check_args
def run_writers(df, asdf=True, cats=None, dirname='/tmp/test',
                str_enc: StrEnc=None, null: bool=False):
    """For given df (should be single column), run series of
    reads/writes
    """
    null_, _ = null_type(df.iloc[:, 0])
    assert null_ == null
    obj_tp = get_obj_type(df) if cats is None else 'infer'
        
    pqr = pq_reader(categories=cats)
    csv_dtype = (None if cats is None else
        dict(zip(cats, it.repeat('category')))
    )
    
    if os.path.exists(dirname):
        print('Deleting', dirname, '... ', end='')
        shutil.rmtree(dirname)
        fprint('Done')
        
    os.mkdir(dirname)
    dir = lambda x: os.path.join(dirname, x)
        
    csv_reader = partial(pd.read_csv, dtype=csv_dtype, index_col=0)
    pq_write_enc = pq_writer(compression='SNAPPY', object_encoding=obj_tp)
    pq_write_len = pq_writer(get_lens=True, compression='SNAPPY')
    
    bc_mkr_mkr = part(mk_bcolz_writer, null=null, str_enc=str_enc)
    blosc_df_wrt = bc_mkr_mkr(method=BEnc.df)
    blosc_uni_wrt = bc_mkr_mkr(method=BEnc.utf8)
    blosc_lst_wrt = bc_mkr_mkr(method=BEnc.list)

    res = {
        'Csv': try_bench(dir('t.csv'), df, DataFrame.to_csv, csv_reader),
        'Bcolz-df': try_bench(dir('t_df.blsc'), df, blosc_df_wrt, read_bcolz),
        'Bcolz-uni': try_bench(dir('t_uni.blsc'), df, blosc_uni_wrt, read_bcolz),
        'Bcolz-lst': try_bench(dir('t_lst.blsc'), df, blosc_lst_wrt, read_bcolz),
        'Feather': try_bench(dir('t.fth'), df, feather.write_dataframe, feather.read_dataframe),
        'Pq-Snappy': try_bench(dir('t_snap.parq'), df, pq_writer(compression='SNAPPY'), pqr),
        'Pq-Snappy-enc': try_bench(dir('t_snap_utf8.parq'), df, pq_write_enc, pqr),
        'Pq-Snappy-ft': try_bench(dir('t_snap_f.parq'), df, pq_write_len, pqr),
        'Pq-Uncompressed': try_bench(dir('t_unc.parq'), df, pq_writer(compression='UNCOMPRESSED'), pqr),
    }
    # try_bench('/tmp/test/t_gzip.parq', df, pq_writer(compression='GZIP'), pqr) + ('Pq-Gzip',),  # <= slow writes
    if asdf:
        return todf(res)
    else:
        return res


# In[ ]:

def run_dfs(base_srs):
    """For base df with single col, generate new columns
    based on original with different ways of encoding
    the string type. Run the full battery of
    read/write benchmarks on each of these new columns.
    """
    d = new_cols(base_srs)
    #global res, resnull, resc, rescnull, resb, resbnull
    res = run_writers(d[['Str']], str_enc=StrEnc.str)
    resnull = run_writers(d[['Str_nls']], str_enc=StrEnc.str, null=True)

    resc = run_writers(d[['Cat']], cats=cols, str_enc=StrEnc.cat)
    rescnull = run_writers(d[['Cat_nls']], cats=cols, str_enc=StrEnc.cat, null=True)

    resb = run_writers(d[['Bytes']], str_enc=StrEnc.byte)
    resbnull = run_writers(d[['Bytes_nls']], str_enc=StrEnc.byte, null=True)
    
    allres = stack_results([
        (res, 'Str', False),
        (resnull, 'Str', True),
        (resc, 'Cat', False),
        (rescnull, 'Cat', True),
        (resb, 'Byte', False),
        (resbnull, 'Byte', True),
    ])
    return allres.assign(Ran=lambda x: ~x.Mb.isnull())


# ## Actually run benchmarks

# In[ ]:

get_ipython().magic('time reslo = run_dfs(df.Low_card)')


# In[ ]:

get_ipython().magic('time resunq = run_dfs(df.Unq)')


# ## Plots

# In[ ]:

_allreslo = reslo.dropna(axis=0)
g = sns.FacetGrid(_allreslo, row='Enc', col='Null', aspect=1.2, size=4)
g.map(plot_scatter, 'Write_time', 'Read_time', 'Mb', 'Fmt')


# In[ ]:

_allres = resunq.dropna(axis=0)
g = sns.FacetGrid(_allres, row='Enc', col='Null', aspect=1.2, size=4)
g.map(part(plot_scatter, sz_fact=10), 'Write_time', 'Read_time', 'Mb', 'Fmt')


# ## Ranking and conclusion

# The following scores the formats based on the time to write and read (but gives twice the weight to read since that's more important to me). For each string encoding setting, it finds the ratio of each format to the one with the best time.
# 
# As an example, consider we're looking for the total weighted time with Non-null byte encoding for just Feather and CSV. If the weighted time for Feather is 2 seconds and for CSV is 4 seconds, then Feather gets a score of 2/2=1 and CSV gets a score of 4/2=2. I then get the median score, aggregating for each of the encoding scenarios.

# In[ ]:

score = lambda df: 1 * df.Write_time + 2 * df.Read_time


def ratio(df):
    s = score(df)
    return s.div(s.min())


def apply_ranking(df):
    ranks = pd.concat([ratio(gdf) for k, gdf in
                       df.groupby(['Enc', 'Null'], sort=False)]
                     ).astype(int)
    df['Ratio'] = ranks
    return df


combine_rankings(reslo, resunq, scoring_func=apply_ranking)


# With these criteria, Feather appears to have consistently good speed across settings, and only fails when writing a bytes column **without** null values ([#283](https://github.com/wesm/feather/issues/283)). The bcolz Dataframe mode does well on the one case where it doesn't take forever to run, and Parquet with snappy encoding and the length setting seems to do well when there are few unique values. It seems that the others which are robust to all the settings have significantly worse performance, though.
# 
# The format that will come closest to Feather for me would probably be Bcolz with the unicode setting (applying `.values.astype('U')` to a frame before saving). It suffers a bit with lots of unique values, but works under *all* of the string and bytes conditions, unlike Feather:

# In[ ]:

combine_rankings(reslo.query('Enc != "Cat"'), resunq.query('Enc != "Cat"'), scoring_func=apply_ranking)


# But despite the occasionaly and unpredictable worst case performance scenarios, the options temporary local DataFrame storage have been improving considerably for python, and there are fewer and fewer reasons to resort to CSV these days.
