import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Summarize UDDA CSV results as mean/std.')
    parser.add_argument('--csv', default='results/param_results.csv')
    parser.add_argument('--out', default='results/param_results_summary.csv')
    parser.add_argument('--final_only', action='store_true', help='Only summarize each run final communication round.')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.final_only:
        df = df[df['comm_round'] == df['main_epochs']]

    group_cols = ['exp_group', 'dataset', 'param_name', 'param_value', 'comm_round']
    summary = (
        df.groupby(group_cols, dropna=False)[['acc', 'nmi', 'ari']]
          .agg(['mean', 'std', 'count'])
          .reset_index()
    )
    summary.columns = ['_'.join([str(x) for x in col if str(x) != '']) for col in summary.columns.values]
    summary.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'Saved summary to {args.out}')
    print(summary.tail(20))


if __name__ == '__main__':
    main()
