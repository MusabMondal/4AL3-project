"""Generate a report table from the recorded development run, without training."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    run=json.loads((ROOT/'results-extension/metrics.json').read_text())
    lines=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\hline',r'Model & F1 & AP & AUC & Brier \\',r'\hline']
    for name,m in run['metrics']['validation'].items():
        lines.append(name.replace('random-forest','Forest').replace('majority','Prior')+' & '+' & '.join(f'{m[k]:.3f}' for k in ['f1','average_precision','roc_auc','brier'])+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\caption{New validation-only comparison on the original seed-0 split. Threshold-dependent scores use $p>0.5$. This run does not evaluate test data.}',r'\label{tab:extension}',r'\end{table}']
    (ROOT/'report/comparison_table.tex').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
