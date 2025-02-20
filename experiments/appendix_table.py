if __name__ == '__main__':
    names_3 = ['Uniform','Normal', 'Uniform and Normal']
    names = ['Standard']*3
    list_1=[]
    dict_translate = {0.8999999999999999: 0.3,
                      0.3: 0.1,
                      1.5: 0.5}
    for i in range(1,4):
        df = pd.read_csv(f"experiment_{i}_results_summary.csv")
        df = df[df['effective_variance']<100]
        df['effective variance limit'] = df['effective variance limit'].apply(lambda x: dict_translate[x])

        df['dataset_name'] = names[i-1]
        df['dataset_name_2'] = names_3[i-1]
        list_1.append(df)
    for i in range(1,3):
        df = pd.read_csv(f"experiment_{i}_27_results_summary.csv")
        df = df[df['effective_variance'] < 100]
        # df['effective variance limit'] = df['effective variance limit'].apply(lambda x: dict_translate[x])

        df['dataset_name'] = names[i - 1]
        df['dataset_name_2'] = names_3[i - 1]
        list_1.append(df)

    names_3  = ['Brownian Motion','Clustered', 'Fractional Brownian Motion']
    names_2 = ['Pathological']*3
    exotic = ['Brownian_Motion.csv','Clustered.csv','Fractional_Brownian_Motion.csv']
    for j,el in enumerate(exotic):
        df = pd.read_csv(el)
        df['effective_variance'] = round(df['effective_variance'],2)
        df = df[df['effective_variance']<100]
        df['dataset_name'] = names_2[j]
        df['dataset_name_2'] = names_3[j]
        list_1.append(df)
    big_df = pd.concat(list_1,ignore_index=True)
    grouped =big_df.groupby(['dataset_name_2','n','small field limit', 'nr of node points', 'effective variance limit'])['time (s)','relative error 2'].mean()
    # print(grouped)
    #
    # print(big_df)
    print(big_df.shape)
