import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
max_flops = 15.7*1e12
if __name__ == '__main__':
    efficiency = []
    n_plot = []
    for n,plt_n in zip([1,10,100,1000],[6,7,8,9]):
        df = pd.read_csv(f'cmake-build-hm_server/{n}m_eff.csv', skiprows=5)
        df = df.pivot(columns ='Metric Description',values='Max').reset_index()
        a = np.nanmax(df['Floating Point Operations(Single Precision)'].values.astype(float))
        b = np.nanmax(df['Floating Point Operations(Single Precision Special)'].values.astype(float))
        # calc_throuput = (a+b)/max_flops
        # print(calc_throuput)
        max_flop_eff = df['FLOP Efficiency(Peak Single)'].dropna().max()
        efficiency.append(float(max_flop_eff.strip('%')))
        n_plot.append(plt_n)

    # print(n_plot)
    # print([round(el,2) for el in efficiency])
    markdown_table = pd.DataFrame([[round(el,2) for el in efficiency]],columns=n_plot)
    markdown_table.index = ['FLOP Efficiency(Peak Single) (%)']
    print(markdown_table.to_latex(escape=False))


    plt.plot(n_plot,efficiency, linestyle='--', marker='o')
    # plt.ylim(0, 20)

    plt.xlabel(r'$\log_{10}(n)$')
    plt.ylabel('FLOP Efficiency(Peak Single) (\%)')
    plt.savefig(f'FLOP_eff.png', bbox_inches='tight')
        # .apply(lambda x: x.strip('%')).values.astype(float)
        # test = df.groupby(['Metric Name'])[['Max','Min','Avg']].max()
        # print(test)