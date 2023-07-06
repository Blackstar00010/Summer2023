
# not used as of 4 July 2023

# Code used to generate placeholder .csv files filled with random real numbers.
# The .csv files generated will soon be replaced by actual data fetched using Eikon Data API, etc.

import numpy as np
import pandas as pd

outputdir = ""

cols = []
for i in range(1, 49):
    cols.append(f"t{i}")
colstr = "absacc	pctacc	acc	aeavol	age	agr	baspread	beta	betasq	bm	bm_ia	cash	cashdebt	cashpr	cfp	cfp_ia	chatoia	turn	chcsho	chempia	hire	chinv	chmom	chpmia	chtx	cinvest	convind	currat	pchcurrat	depr	divi	divo	dy	dolvol	ear	egr	ep	gma	herf	idiovol	ill	indmom	invest	IPO	lev	lgr	maxret	ms	ps	mve	mve_ia	nincr	operprof	pchcapx_ia	pchdepr	pchgm_pchsale	pchsale_pchrect	pricedelay	quick	pchquick	rd	retvol	roaq	roeq	roic	rsup	salecash	salerec	securedind	sgr	sin	SP	std_dolvol	std_turn	sue	tang	tb	zerotrade"
colstr = colstr.split("\t")
for item in colstr:
    cols.append(item)

rows = []
for i in range(1, 1001):
    rows.append(f"firm{i}")

# """
for yr in range(1990, 2023):
    for mm in range(1, 13):
        num_rows = len(rows)
        num_cols = len(cols)

        random_data = np.random.uniform(-10, 10, size=(num_rows, num_cols))
        df = pd.DataFrame(random_data, index=rows, columns=cols)
        df.to_csv(f"./files/feature_set/{yr}-{mm}.csv")
# """

