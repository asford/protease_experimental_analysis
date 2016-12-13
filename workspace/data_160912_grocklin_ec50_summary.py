import pandas
import numpy
from os import path

__all__ = ["scramble_data"]

scramble_data = (
    pandas.read_csv(path.join(path.dirname(__file__), "160912_grocklin_data.txt"), delim_whitespace=True)
    .query("data0 > -4.5")
    .query("designtype == 'scramble'")
    [["name", "run", "sequence", "protease", "ec50"]]
    .groupby(["sequence", "protease"])
    .apply(lambda d: d.sort_values(by="ec50", ascending=False).head(1)))

scramble_data["ec50"] = numpy.clip(scramble_data["ec50"], 0, None)

scramble_data = scramble_data[[ "C" not in s for s in scramble_data["sequence"]]]

scramble_data["full_sequence"] = "ASHM" + scramble_data["sequence"] + "LEGG"
scramble_data["round"] = scramble_data["run"].str.extract("([^_]*)_.*")