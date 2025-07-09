import pytest
import numpy as np

from StarStream import KernelPDF

def test_KernelPDF():
    data = np.array([
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
        [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    ])
    hs = np.repeat(
        [[0.1, 0.2, 0.1, 0.2, 0.1, 0.2]], repeats=len(data), axis=1
    )

    spacings = np.array([0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
    grids = [np.arange(0.0,1.0+0.01,spacing) for spacing in spacings]
    x0, x1, x2, x3, x4, x5 = np.meshgrid(*grids)
    data_est = np.c_[
        x0.flatten(), x1.flatten(), x2.flatten(),
        x3.flatten(), x4.flatten(), x5.flatten()
    ]

    for gs in [[None]*6, grids]:
        pdf = KernelPDF(data, gs, hs, [[0, 1, 2, 3, 4, 5]])
        integral = np.sum(pdf.eval_pdf(data_est)) * np.prod(spacings)
        assert integral == pytest.approx(1.0, 0.1)
