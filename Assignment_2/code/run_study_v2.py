import argparse
from study_v2 import run_study
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--config',required=True); p.add_argument('--output',required=True); p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1']); a=p.parse_args()
    run_study(a.data,a.config,a.output,a.devices)
