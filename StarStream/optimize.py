# Licensed under BSD-3-Clause License - see LICENSE

import numpy as np

__all__ = [
    "optimize_mle"
]

def optimize_mle(data, err, pdf_signal, pdf_bg, frac_list=None):
	"""
	data: array-like (N, M): N data points with M dimensions
	err: array-like (N, M): N data points with M dimensions
	frac_list: list of fractions for evaluating lnL, default: None
	pdf_signal, pdf_bg: KernelPDF objects
	"""

	prob_signal = pdf_signal.eval_pdf(data, err_eval=err)
	prob_bg = pdf_bg.eval_pdf(data, err_eval=err)
	prob_bg[prob_bg==0] = np.min(prob_bg[prob_bg!=0])

	lnL_base = np.sum(np.log(prob_bg)) # no signal

	if frac_list is None:
		frac_list = 10.0**np.arange(-6.0, -1.0+0.001, 0.02)

	lnL_list = []
	for frac in frac_list:
	    lnLi = np.log(prob_signal*frac + prob_bg*(1-frac))
	    lnL_list.append(np.sum(lnLi))
	lnL_list = np.array(lnL_list)

	dlnL_list = lnL_list-lnL_base

	fbest = frac_list[np.argmax(dlnL_list)]
	dlnL_max = np.max(dlnL_list)

	if dlnL_max <=0:
		fbest = 0.0

	prob = fbest*prob_signal / (fbest*prob_signal+(1-fbest)*prob_bg)

	return fbest, prob, frac_list, dlnL_list