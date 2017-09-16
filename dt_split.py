#!/usr/bin/env python
#
# Data splitting tool, capable of handling multiple files at once.
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev

from __future__ import print_function
import sys
import argparse
import os
import random as rnd

try:
    from itertools import izip
except ImportError:
    izip = zip

# Deal with the arguments.
argp = argparse.ArgumentParser(
    description="A data splitting tool, capable of dealing with several times simultaneously. "
                "Each of the given files is split in number of files corresponding to the given list "
                "of probabilities with suffixes added according to the list of names.",
    epilog="No loading in memory happens, but just a single line from each of the given "
           "data files at every step, which is then distributed to either of the specified splits. "
           "The numbers provided for the splitting can be seen as a ratio, however what happens is "
           "that a long array of size, the sum of all given numbers is built and shuffled. Then "
           "during input file(s) iteration this array is walked in round-robin manner determining the "
           "split and re-shuffling it every time its end is reached."
)
argp.add_argument('data', type=str, nargs='+',
                  help="The list of data files to be split.")
argp.add_argument('-n', '--number', type=int, dest="count", nargs='+',
                  help="The list of counts per split to put.")
argp.add_argument('-s', '--suffix', type=str, dest="suffix", nargs='+',
                  help="The list of suffixes to be added to each of the splits, based on the counts given.")

argp.add_argument('-q', '--quiet', required=False, action="store_true", dest="quite",
                  help="Whether to suppress the more detailed info messages.")

cmd_args = argp.parse_args()


def _log(*args, **kwargs):
    if not cmd_args.quite:
        print(*args, file=sys.stderr, **kwargs)

# Some sanity checks
l_count = len(cmd_args.count)
assert l_count > 1
assert len(cmd_args.suffix) == l_count

# Prepare the index list
pos_idx = []
for c, i in zip(cmd_args.count, range(l_count)):
    pos_idx.extend([i] * c)

rnd.shuffle(pos_idx)
l_count = len(pos_idx)
# Now prepare the files handles
l_data = len(cmd_args.data)
outf = []
inf = []
for i_name in cmd_args.data:
    _log ("Preparing for split: %s ..." % i_name)
    inf.append(open(i_name, "rt"))
    base, ext = os.path.splitext(i_name)
    splits = [open(base + suf + ext, "wt") for suf in cmd_args.suffix]
    outf.append(splits)


# Go with the actual processing
try:
    seq = 0
    for lines in izip(*inf):
        split = pos_idx[seq]
        seq += 1
        if seq >= l_count:
            # Reshuffle, so there is not pattern in the data splitting
            rnd.shuffle(pos_idx)
            seq = 0
        for l, i in zip(lines, range(l_data)):
            outf[i][split].write(l)

except IOError as e:
    _log(e)
finally:
    # Need to close everybody
    for f in inf:
        f.close()
    for o_list in outf:
        for f in o_list:
            f.close()
    _log("Done.")