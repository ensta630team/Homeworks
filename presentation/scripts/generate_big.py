import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from source.data.loaders import create_dataset
from source.fit import inference as testmodule
from source.display.hw import hw3 as plot_utils
from source.display.hw.hw1 import save_figure


# En cada iteracion simulamos un proceso estocastico y_t = y_{t-1} + u_t
dataset = create_dataset('./data/base_25.xls', problem=6)

test_results=[]
for lab in ['y', 'p', 'dtp', 'dty', 'i']:
    try:
        # metrics_0 = testmodule.test_df(dataset[lab])
        metrics_1 = testmodule.test_adf(dataset[lab], max_k=10, model_type='c', alpha=0.05)
        metrics_2 = testmodule.test_pp(dataset[lab])
        metrics_3 = testmodule.test_kpss(dataset[lab])
        metrics_4 = testmodule.test_perron(dataset[lab], varname=lab, breakpoint=50)
        metrics_5 = testmodule.test_zivot_andrews_v2(dataset[lab])
        metrics_6 = testmodule.test_bierens(dataset[lab], max_k=5, max_m=3, alpha=0.05)
        metrics_7 = testmodule.test_murray_nelson(dataset[lab], periodo=[10, 15], k=5)
        metrics_8 = testmodule.test_gls(dataset[lab], p=13)

        columns = []
        for j, m in enumerate([metrics_1, metrics_2, metrics_3, metrics_4, metrics_5,  metrics_7]):
            columns.append(m[['model', 'p-value']])
        columns = pd.concat(columns, axis=0, ignore_index=True)

        columns.to_csv('./presentation/backup/bitreal_{}.csv'.format(lab), index=False)

    except Exception as e:
        print(e)
        continue





