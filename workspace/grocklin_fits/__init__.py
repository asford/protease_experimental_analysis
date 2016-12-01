import os
from os import path
from glob import glob

import pandas

infiles = glob(path.join(path.dirname(__file__), "*.output"))

infiles = pandas.Series(list(infiles))

fits = {
    n : pandas.read_table(f, delim_whitespace=True)
    for n, f in zip(infiles.str.extract(".*/(.*)\.sel", expand=False), infiles)
}
