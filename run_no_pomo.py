# run_no_pomo.py

import sys

from experiment import main


def _strip_ablation_args(argv):
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == '--ablation':
            skip_next = True
            continue
        if arg.startswith('--ablation='):
            continue
        cleaned.append(arg)
    return cleaned


def run():
    argv = _strip_ablation_args(sys.argv[1:])
    main(['--ablation', 'no_pomo', '--single_rollout', *argv])


if __name__ == '__main__':
    run()
