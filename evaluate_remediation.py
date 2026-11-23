"""GraphML-CloudSecurity: held-out remediation evaluation.

Run after the 100-scenario pilot. Excludes those same 100 scenarios using
exactly the original sorted-then-seeded-shuffle selection. Uses 2,900 remaining
scenarios from the existing model test split. No real AWS modifications.
"""
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from counterfactual_remediation import (
    load_model, load_graph, find_attack_paths, generate_actions,
    apply_intervention, predict_graph_risk,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'datasets'
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'independent_remediation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
PILOT_SCENARIOS = 100
MAX_EVALUATION_SCENARIOS = None  # None means all remaining scenarios (normally 2,900)
BUDGETS = [2, 5, 10, 20]
STRATEGIES = ['severity', 'graphsage', 'graph_rules', 'hybrid_graphsage']
STRATEGY_NAMES = {
    'severity': 'Severity Baseline',
    'graphsage': 'GraphSAGE',
    'graph_rules': 'Greedy Graph Rules',
    'hybrid_graphsage': 'Hybrid GraphSAGE',
}


def simulator_metrics(graph):
    paths = find_attack_paths(graph)
    return {
        'paths': len(paths),
        'assets': len({path[-1] for path in paths}),
        'risky_nodes': len({node for path in paths for node in path[1:]}),
    }


def ai_risk(graph, model, device, threshold):
    predictions = predict_graph_risk(graph, model, device, threshold)
    return sum(item['risk_score'] for item in predictions.values())


def action_key(action):
    return (action['action_type'], action.get('source', ''), action['target'])


def select_action(graph, actions, strategy, model, device, threshold):
    if not actions:
        return None

    if strategy == 'severity':
        patches = [a for a in actions if a['action_type'] == 'patch_vulnerability']
        if not patches:
            return None
        return max(patches, key=lambda a: graph.nodes[a['target']]['severity'])

    if strategy not in ('graphsage', 'graph_rules', 'hybrid_graphsage'):
        raise ValueError(f'Unknown strategy: {strategy}')

    before_rules = simulator_metrics(graph)
    before_ai = (
        ai_risk(graph, model, device, threshold)
        if strategy in ('graphsage', 'hybrid_graphsage') else None
    )
    best_action, best_score = None, None

    for action in actions:
        cost = float(action['cost'])
        if cost <= 0:
            continue
        modified = apply_intervention(graph, action)

        if strategy == 'graphsage':
            reduction = before_ai - ai_risk(modified, model, device, threshold)
            if reduction <= 1e-8:
                continue
            score = (reduction / cost,)
        else:
            after_rules = simulator_metrics(modified)
            assets_removed = before_rules['assets'] - after_rules['assets']
            paths_removed = before_rules['paths'] - after_rules['paths']

            if strategy == 'graph_rules':
                value = (assets_removed + 0.001 * paths_removed) / cost
                if value <= 1e-8:
                    continue
                score = (value,)
            else:
                # Verification oracle: synthetic path rules also define labels.
                if assets_removed <= 0 and paths_removed <= 0:
                    continue
                reduction = before_ai - ai_risk(modified, model, device, threshold)
                score = (reduction / cost, assets_removed / cost, paths_removed / cost)

        if best_score is None or score > best_score:
            best_action, best_score = action, score

    return best_action


def optimize(graph, strategy, budget, model, device, threshold):
    start = time.time()
    current = graph.copy()
    initial = simulator_metrics(current)
    remaining = float(budget)
    applied = []
    used = set()

    while remaining > 0:
        if strategy in ('graph_rules', 'hybrid_graphsage'):
            if simulator_metrics(current)['paths'] == 0:
                break

        affordable = [
            a for a in generate_actions(current)
            if 0 < float(a['cost']) <= remaining and action_key(a) not in used
        ]
        if not affordable:
            break

        action = select_action(current, affordable, strategy, model, device, threshold)
        if action is None:
            break

        current = apply_intervention(current, action)
        remaining -= float(action['cost'])
        used.add(action_key(action))
        applied.append(action)

    final = simulator_metrics(current)
    return {
        'strategy': strategy,
        'budget': budget,
        'cost_used': round(float(budget) - remaining, 6),
        'actions_applied': len(applied),
        'paths_before': initial['paths'],
        'paths_after': final['paths'],
        'paths_removed': max(0, initial['paths'] - final['paths']),
        'assets_before': initial['assets'],
        'assets_after': final['assets'],
        'assets_protected': max(0, initial['assets'] - final['assets']),
        'risky_nodes_reduced': max(0, initial['risky_nodes'] - final['risky_nodes']),
        'complete_remediation': int(initial['paths'] > 0 and final['paths'] == 0),
        'execution_time_seconds': round(time.time() - start, 6),
        'selected_action_details': json.dumps(applied),
    }


def aggregate_results(frame):
    rows = []
    for budget in BUDGETS:
        for strategy in STRATEGIES:
            sub = frame[(frame['budget'] == budget) & (frame['strategy'] == strategy)]
            if sub.empty:
                continue
            risky = sub[sub['paths_before'] > 0]
            before_paths = int(sub['paths_before'].sum())
            removed_paths = int(sub['paths_removed'].sum())
            before_assets = int(sub['assets_before'].sum())
            protected_assets = int(sub['assets_protected'].sum())
            total_cost = float(sub['cost_used'].sum())
            rows.append({
                'strategy': strategy,
                'budget': budget,
                'environments': len(sub),
                'risky_environments': len(risky),
                'total_paths_before': before_paths,
                'total_paths_removed': removed_paths,
                'path_reduction_percentage': round(100 * removed_paths / before_paths, 4) if before_paths else 0.,
                'total_assets_before': before_assets,
                'total_assets_protected': protected_assets,
                'asset_protection_percentage': round(100 * protected_assets / before_assets, 4) if before_assets else 0.,
                'complete_remediation_percentage': round(100 * risky['complete_remediation'].mean(), 4) if len(risky) else 0.,
                'total_cost': round(total_cost, 4),
                'average_cost_per_environment': round(float(sub['cost_used'].mean()), 4),
                'assets_protected_per_cost_unit': round(protected_assets / total_cost, 4) if total_cost else 0.,
                'average_runtime_seconds': round(float(sub['execution_time_seconds'].mean()), 6),
            })
    return pd.DataFrame(rows)


def plot_results(summary):
    plots = [
        ('path_reduction_percentage', 'Attack Path Reduction', 'Attack paths removed (%)', 'budget_path_reduction.png'),
        ('asset_protection_percentage', 'Critical Asset Protection', 'Critical assets protected (%)', 'budget_asset_protection.png'),
        ('complete_remediation_percentage', 'Complete Remediation', 'Risky environments fully remediated (%)', 'budget_complete_remediation.png'),
        ('assets_protected_per_cost_unit', 'Remediation Cost Efficiency', 'Assets protected per cost unit', 'budget_cost_efficiency.png'),
    ]
    for metric, title, ylabel, filename in plots:
        plt.figure(figsize=(10, 6))
        for strategy in STRATEGIES:
            sub = summary[summary['strategy'] == strategy].sort_values('budget')
            plt.plot(sub['budget'], sub[metric], marker='o', linewidth=2, label=STRATEGY_NAMES[strategy])
        plt.title(title)
        plt.xlabel('Remediation budget (hypothetical cost units)')
        plt.ylabel(ylabel)
        plt.xticks(BUDGETS)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()


def select_evaluation_files():
    manifest_path = DATASET_DIR / 'test' / 'manifest.json'
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)

    # Exactly mirrors the pilot selection procedure.
    shuffled = sorted(manifest['files'])
    random.Random(SEED).shuffle(shuffled)
    if len(shuffled) <= PILOT_SCENARIOS:
        raise ValueError('Test manifest does not contain enough scenarios after pilot exclusion.')
    pilot = set(shuffled[:PILOT_SCENARIOS])
    evaluation = shuffled[PILOT_SCENARIOS:]
    if MAX_EVALUATION_SCENARIOS is not None:
        evaluation = evaluation[:MAX_EVALUATION_SCENARIOS]
    if pilot.intersection(evaluation):
        raise AssertionError('Pilot and evaluation scenarios overlap.')
    return sorted(pilot), evaluation


def main():
    start = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    print('=' * 70)
    print('HELD-OUT HYBRID GRAPHSAGE REMEDIATION EVALUATION')
    print('=' * 70)
    model, device, threshold = load_model()
    pilot_files, evaluation_files = select_evaluation_files()
    expected = len(evaluation_files) * len(STRATEGIES) * len(BUDGETS)
    print(f'Device: {device} | Threshold: {threshold}')
    print(f'Excluded pilot scenarios: {len(pilot_files)}')
    print(f'Evaluation scenarios: {len(evaluation_files)}')
    print(f'Budgets: {BUDGETS} | Strategies: {STRATEGIES}')
    print(f'Planned evaluations: {expected}')
    print(f'Output directory: {OUTPUT_DIR}')

    # Write the exact scenario allocation for reproducibility.
    with open(OUTPUT_DIR / 'scenario_selection.json', 'w', encoding='utf-8') as handle:
        json.dump({
            'random_seed': SEED,
            'selection_method': 'sorted test manifest, random.Random(42).shuffle, exclude first 100',
            'pilot_scenarios': pilot_files,
            'evaluation_scenarios': evaluation_files,
            'pilot_evaluation_overlap': False,
        }, handle, indent=2)

    all_results = []
    for index, filename in enumerate(evaluation_files, start=1):
        graph = load_graph(DATASET_DIR / 'labeled' / filename)
        for budget in BUDGETS:
            for strategy in STRATEGIES:
                row = optimize(graph, strategy, budget, model, device, threshold)
                row['scenario'] = filename
                all_results.append(row)

        if index % 50 == 0 or index == len(evaluation_files):
            pd.DataFrame(all_results).to_csv(OUTPUT_DIR / 'remediation_progress.csv', index=False)
            print(f'Processed {index}/{len(evaluation_files)} scenarios; elapsed {time.time()-start:.1f}s', flush=True)

    if len(all_results) != expected:
        raise AssertionError('Incorrect number of evaluations.')

    frame = pd.DataFrame(all_results)
    frame.to_csv(OUTPUT_DIR / 'large_scale_remediation.csv', index=False)
    summary = aggregate_results(frame)
    summary.to_csv(OUTPUT_DIR / 'remediation_budget_summary.csv', index=False)
    plot_results(summary)
    elapsed = time.time() - start

    report = {
        'project': 'GraphML-CloudSecurity',
        'source': 'synthetic',
        'experiment': 'held_out_remediation_evaluation',
        'optimization_method': 'greedy_sequential',
        'random_seed': SEED,
        'pilot_scenarios_excluded': len(pilot_files),
        'environments': len(evaluation_files),
        'budgets': BUDGETS,
        'strategies': STRATEGIES,
        'total_experiments': len(all_results),
        'simulation_only': True,
        'cost_units': 'hypothetical',
        'methodological_note': (
            'These 2,900 scenarios were held out from remediation strategy development, '
            'but the same 3,000-scenario test split was previously used to evaluate GraphSAGE. '
            'The graph-rule and hybrid strategies use the simulator that defines the synthetic attack paths. '
            'Severity is a patch-only baseline with a different action space.'
        ),
        'execution_time_seconds': round(elapsed, 4),
        'results': summary.to_dict(orient='records'),
    }
    with open(OUTPUT_DIR / 'large_scale_remediation_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    print('\n' + '=' * 70)
    print('HELD-OUT REMEDIATION COMPARISON')
    print('=' * 70)
    columns = [
        'strategy', 'budget', 'path_reduction_percentage',
        'asset_protection_percentage', 'complete_remediation_percentage',
        'assets_protected_per_cost_unit',
    ]
    print(summary[columns].to_string(index=False))
    print(f'\nExecution time: {elapsed:.2f} seconds')
    print(f'Results saved to: {OUTPUT_DIR}')
    print('IMPORTANT: Synthetic attack paths, hypothetical costs, no real AWS changes.')


if __name__ == '__main__':
    main()
