"""Grade experiment responses with the LLM judges locally, after the Modal GPU run.

The Modal entrypoints run generation only (judges='none'), so GPU time isn't spent
waiting on OpenAI. Download the results, then fill in the judge columns here:

    modal volume get introspection-results new_results . --force
    python judge_results.py --csv new_results/output_coherence_affirmation.csv

Requires OPENAI_API_KEY (the same one used on Modal). Judge calls run concurrently
(--workers) and the CSV is checkpointed as grading progresses, so re-running is safe:
only rows whose judge columns are still empty get graded. Finishes by writing the
Parquet copy and the plot, mirroring what main.py produces when judges run inline.

The coherence/affirmative judges grade against the full chat-templated conversation,
which main.py saves once per run as new_results/judge_question_<run_name>.txt (it is
identical for every row of a run). Point --judge_question_file elsewhere to override.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from tqdm import tqdm
from api_utils import query_llm_judge

# Same trial question main.py passes to the word-identification judges
QUESTION_TEXT = "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"

# Judges gradable from the CSV alone. mcq_correct / injection_strength_correct need
# per-row context (options, expected category) that isn't fully stored, so those runs
# should keep judging inline in main.py.
JUDGES = ['coherence', 'affirmative_response', 'thinking_about_word',
          'affirmative_response_followed_by_correct_identification']

# Judges that grade against the full conversation (judge_question file) rather than
# the bare trial question + concept word
CONVERSATION_JUDGES = {'coherence', 'affirmative_response'}


def grade_row(row, judge, judge_question):
    response = '' if pd.isna(row['response']) else str(row['response'])
    if judge in CONVERSATION_JUDGES:
        return query_llm_judge(question=judge_question, response=response, grading_type=judge)
    return query_llm_judge(question=QUESTION_TEXT, response=response, word=row['concept'], grading_type=judge)


def to_rate(series):
    """Judge column values as 0/1 floats (NaN for ungraded), robust to CSV round-trips."""
    return series.map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0, 1.0: 1.0, 0.0: 0.0})


# Okabe-Ito, colorblind-safe, assigned to coeff in fixed order (never cycled); matches
# plot_projection.py's convention. Identity is also carried by the legend, never color alone.
COEFF_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
# One (linestyle, marker, label) per judge, so a judge looks the same at every coeff:
# detection solid, coherence dashed. Marker shape doubles the linestyle cue.
JUDGE_STYLES = {
    'affirmative_response': ('-', 'o', 'Injection reported'),
    'coherence': ('--', 's', 'Coherent response'),
    'affirmative_response_followed_by_correct_identification': ('-.', '^', 'correctly identified'),
    'thinking_about_word': (':', 'D', 'thinking about word'),
}


def make_plot(df, judges, run_label, plots_dir=Path('plots')):
    layers = sorted(df['layer'].unique())
    coeffs = sorted(df['coeff'].unique())

    fig, ax = plt.subplots(figsize=(10, 6.5))
    per_cell_n = 0
    for ci, coeff in enumerate(coeffs):
        color = COEFF_COLORS[ci % len(COEFF_COLORS)]
        for judge in judges:
            linestyle, marker, _ = JUDGE_STYLES.get(judge, ('-', 'o', judge))
            rates = to_rate(df[df['coeff'] == coeff][f'{judge}_judge'])
            grouped = rates.groupby(df['layer']).agg(['mean', 'count'])
            grouped = grouped[grouped['count'] > 0]
            if grouped.empty:
                continue
            per_cell_n = max(per_cell_n, int(grouped['count'].max()))
            y_err = (grouped['mean'] * (1 - grouped['mean']) / grouped['count']) ** 0.5
            ax.errorbar(grouped.index, grouped['mean'], yerr=y_err, color=color,
                        linestyle=linestyle, marker=marker, linewidth=2, markersize=7,
                        capsize=3, zorder=3)

    # Two orthogonal legends instead of a coeff-x-judge cross product: color carries the
    # coeff, linestyle+marker carry the judge (drawn in neutral gray so no coeff is implied).
    coeff_handles = [Line2D([], [], color=COEFF_COLORS[i % len(COEFF_COLORS)], linewidth=2.5,
                            label=f'coeff={coeff:g}') for i, coeff in enumerate(coeffs)]
    judge_handles = [Line2D([], [], color='#555555', linewidth=2, markersize=7,
                            linestyle=JUDGE_STYLES.get(j, ('-', 'o', j))[0],
                            marker=JUDGE_STYLES.get(j, ('-', 'o', j))[1],
                            label=JUDGE_STYLES.get(j, ('-', 'o', j))[2]) for j in judges]
    coeff_legend = ax.legend(handles=coeff_handles, title='Injection coefficient', fontsize=9,
                             loc='lower left', framealpha=0.9)
    ax.add_artist(coeff_legend)
    ax.legend(handles=judge_handles, title='judge', fontsize=9, loc='lower right',
              framealpha=0.9)

    temperature = df['temperature'].iloc[0] if 'temperature' in df else '?'
    experiment_type = df['type'].iloc[0] if 'type' in df else '?'
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Rate (mean over concepts x trials)', fontsize=11)
    ax.set_title('Configuration sweep for coherent injection detection', fontsize=12)
    # Run details stay out of the figure; printed for use in captions
    print(f"Run: {run_label} (type={experiment_type}, temp={temperature}, n={per_cell_n}/cell)")
    if len(layers) == 1:
        ax.set_xlim(layers[0] - 1, layers[0] + 1)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()

    plots_dir.mkdir(exist_ok=True)
    figure_path = plots_dir / f'main_figure_{run_label}.png'
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {figure_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fill in LLM judge columns of a results CSV (run locally after Modal)")
    parser.add_argument("--csv", type=str, required=True,
                        help="Results CSV from main.py, e.g. new_results/output_coherence_affirmation.csv")
    parser.add_argument("--judges", type=str, nargs="+", default=["coherence", "affirmative_response"],
                        choices=JUDGES,
                        help="Judges to run on rows where their column is still empty "
                             "(default: coherence affirmative_response)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent judge API calls (default: 8)")
    parser.add_argument("--judge_question_file", type=str, default=None,
                        help="Conversation text for the coherence/affirmative judges. Defaults to "
                             "judge_question_<run_name>.txt next to the CSV (written by main.py)")
    parser.add_argument("--no_plot", action="store_true", help="Skip writing the plot")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save the CSV after this many completed judge calls (default: 50)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    run_label = csv_path.stem.removeprefix('output_')
    df = pd.read_csv(csv_path)
    for judge in args.judges:
        # An ungraded column reads back as all-NaN float64; keep it object so writing
        # bools stores True/False (matching main.py's format), not coerced 1.0/0.0
        df[f'{judge}_judge'] = df[f'{judge}_judge'].astype(object)

    # Only grade rows whose judge column is still empty, so re-runs resume where they left off
    pending = [(idx, judge)
               for judge in args.judges
               for idx in df.index[to_rate(df[f'{judge}_judge']).isna()]]
    print(f"{len(df)} rows; {len(pending)} judge calls to make "
          f"({', '.join(args.judges)}), {args.workers} concurrent")

    judge_question = None
    if CONVERSATION_JUDGES & {judge for _, judge in pending}:
        question_path = Path(args.judge_question_file) if args.judge_question_file \
            else csv_path.parent / f'judge_question_{run_label}.txt'
        if not question_path.exists():
            raise FileNotFoundError(
                f"{question_path} not found. main.py writes it next to the CSV during the run; "
                f"download it with the results, or pass --judge_question_file")
        judge_question = question_path.read_text(encoding='utf-8')

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(grade_row, df.loc[idx], judge, judge_question): (idx, judge)
                   for idx, judge in pending}
        for future in tqdm(as_completed(futures), total=len(pending), desc="Judging"):
            idx, judge = futures[future]
            # None (API error / unclear verdict) stays empty and is retried on the next run
            verdict = future.result()
            if verdict is not None:
                df.at[idx, f'{judge}_judge'] = verdict
            done += 1
            if done % args.checkpoint_every == 0 or done == len(pending):
                df.to_csv(csv_path, index=False)
                print(f"[{done}/{len(pending)}] checkpointed {csv_path}")

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    try:
        parquet_path = csv_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False)
        print(f"Results saved to {parquet_path}")
    except ImportError:
        print("Note: pyarrow not installed, skipping Parquet export. Install with: pip install pyarrow")

    # Summary rates per (layer, coeff)
    for judge in args.judges:
        rates = to_rate(df[f'{judge}_judge'])
        summary = rates.groupby([df['layer'], df['coeff']]).mean().unstack()
        print(f"\n{judge} rate (rows: layer, cols: coeff):\n{summary.round(3)}")

    if not args.no_plot:
        make_plot(df, args.judges, run_label)


if __name__ == "__main__":
    main()
