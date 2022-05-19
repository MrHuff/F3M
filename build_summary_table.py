import pandas as pd
import numpy as np


def poly_fit(d,big_df):
    subset = big_df[big_df['d']==d]

    big_X = subset['n'].values
    big_Y = subset['time (s)'].values
    m, b = np.polyfit(np.log10(big_X), np.log10(big_Y), 1)
    return m

def load_data(d,ablation):
    if d==3:
        names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
        names = ['Standard'] * 3
        list_1 = []
        for i in range(1, 4):
            df = pd.read_csv(f"experiment_{i}_results_summary_ablation.csv") if ablation else pd.read_csv(f"experiment_{i}_results_summary.csv")
            df['dataset_name'] = names[i - 1]
            df['dataset_name_2'] = names_3[i - 1]
            list_1.append(df)
        names_3 = ['Brownian Motion', 'Clustered', 'Fractional Brownian Motion']
        names_2 = ['Pathological'] * 3
        exotic = ['Brownian_ablation.csv', 'Clustered_ablation.csv', 'Fractional_Brownian_Motion_ablation.csv'] if ablation else ['Brownian_Motion.csv', 'Clustered.csv', 'Fractional_Brownian_Motion.csv']
        for j, el in enumerate(exotic):
            df = pd.read_csv(el)
            df['effective_variance'] = round(df['effective_variance'], 2)
            df['dataset_name'] = names_2[j]
            df['dataset_name_2'] = names_3[j]
            list_1.append(df)
        big_df = pd.concat(list_1, ignore_index=True)
    else:
        names_3 = ['Uniform','Normal', 'Uniform and Normal']
        list_1 = []
        for i in range(6, 9):
            df = pd.read_csv(f"exp{i}_ablation_summary.csv", index_col=0) if ablation else pd.read_csv(f"experiment_{i}_results_summary.csv",index_col=0)
            df = df[df['d'] == d]
            df['dataset_name'] = names_3[i - 6]
            df['dataset_name_2'] = names_3[i - 6]
            list_1.append(df)
        big_df = pd.concat(list_1, ignore_index=True)
        big_df = big_df[big_df['dataset_name'].isin(['Uniform', 'Uniform and Normal'])]
    mask = (big_df['relative error 2']<=1e-2) & (big_df['relative error 2']>1e-6)
    big_df = big_df[mask]
    return big_df
if __name__ == '__main__':

    df_3 = load_data(3, False)
    df_4 = load_data(4, False)
    df_5 = load_data(5, False)
    df_high = pd.read_csv(f"larger_dims.csv", index_col=0)
    df_high = df_high[df_high['relative error 2']<0.1]

    df_low = pd.read_csv(f"lower_dims.csv", index_col=0)
    df_low = df_low[(df_low['relative error 2']<=0.01)&(df_low['relative error 2']>1e-6)]

    cols = ['dataset_name', 'n', 'd', 'nr of node points', 'small field limit',
       'effective_variance', 'effective variance limit', 'relative error 2',
       'time (s)']

    big_df = pd.concat([df_low,df_high,df_5[cols],df_4[cols],df_3[cols]],axis=0).reset_index(drop=True)
    estimated_complexity = {}
    for d in big_df['d'].unique().tolist():
        m = poly_fit(d,big_df)
        estimated_complexity[d] = round(m,2)

    print(estimated_complexity)
    a = big_df.groupby(['n','d'])['time (s)','relative error 2'].mean().reset_index()
    b = big_df.groupby(['n','d'])['time (s)','relative error 2'].std().reset_index()
    a['time (s)'] = a['time (s)'].apply(lambda x:  '$\makecell{'+str(round(x,1))+r'\\ \pm') +  b['time (s)'].apply(lambda x: str(round(x,1))+'}$' )
    a['relative error 2'] = a['relative error 2'].apply(lambda x:'$\makecell{' +str(round(x,4))+r'\\ \pm') +  b['relative error 2'].apply(lambda x: str(round(x,4))+'}$' )

    d = a.pivot(index='n',columns='d',values=['time (s)','relative error 2'])
    d.to_csv('res_table_neurips22.csv')

