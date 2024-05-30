from argparse import ArgumentParser
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == '__main__':
    parser = ArgumentParser()

    args = parser.parse_args(description='Generate multisite, multivariable simulations')

    parser.add_argument('--file_in_path',
                        dest='file_in_path', required=True, type=str,
                        help='path to the .csv observed data file.')
    parser.add_argument('--start_year',
                        dest="start_year",
                        required=True, type=int,
                        help='first year of simulation.')
    parser.add_argument('--n_years',
                        dest='n_years',
                        default='1', type=int,
                        help='number of years to be simulated (including the start_year as the first one).')
    parser.add_argument('--n_simulations',
                        dest='n_simulations', type=int,
                        default="1",
                        help='number that each year will be simulated.')
    parser.add_argument('--wet_extreme_quantile_threshold',
                        dest='wet_extreme_quantile_threshold', type=float,
                        default="0.999",
                        help='extreme-wet quantile to be applyed monthly.')
    parser.add_argument('--use_g2s',
                        dest='use_g2s', type=str2bool,
                        default=False,
                        help='flag to determine whether to use G2S for spatial variability enhancement.')

    args = parser.parse_args()
    print(args)


