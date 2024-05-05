import pandas as pd


def make_epoch_table_1d(history_all, decimals=1):
    table = (
        history_all.groupby(['weight_type', 'seed'])
        .max()
        .groupby('weight_type')
        .agg(
            Mean=pd.NamedAgg('epoch', 'mean'),
            Std=pd.NamedAgg('epoch', 'std'),
        )
        .round(decimals)
        .reindex(['random', '1', '10', '20', '100'])
        .rename(index={
            'random': 'Random',
            '1': 'Pre-train 1',
            '10': 'Pre-train 10',
            '20': 'Pre-train 20',
            '100': 'Pre-train 100',
        })
        .reset_index()
        .rename(columns={
            'weight_type': 'Scenario',
        })
    )
    # Explicitly select columns in desired order.
    table = table[['Scenario', 'Mean', 'Std']]
    return table
