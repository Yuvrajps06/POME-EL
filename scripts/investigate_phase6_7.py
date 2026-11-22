import sys
from pathlib import Path
sys.path.insert(0, '/home/yuvrajps/projects/POME/PricePilot')

import numpy as np
import pandas as pd
from src.experiments.comparison import run_benchmark_experiments, run_strategy_trajectory
from src.experiments.simulation import MarketEnvironment
from src.competitor.response_model import load_competitor_model

def run_investigation():
    b = 79.18
    c = 0.50
    a = 165.39
    cost = 0.83
    initial_p_l = 1.38
    initial_p_f = 1.37
    n_sims = 100
    n_periods = 28
    
    print("=" * 70)
    print("STEP 1: 100-Run Evaluation of A, B, C under Phase 6 defaults")
    print("=" * 70)
    
    summary_df, traj_df, sim_runs_df = run_benchmark_experiments(
        initial_leader_price=initial_p_l,
        initial_follower_price=initial_p_f,
        cost=cost,
        a=a,
        b=b,
        c=c,
        competitor_true_behavior="adaptive",
        n_periods=n_periods,
        n_simulations=n_sims,
        rule_behavior="match",
        master_seed=42,
    )
    
    # Calculate distribution
    stats_rows = []
    for s_col, s_label in [('profit_A', 'Strategy A (Baseline)'), ('profit_B', 'Strategy B (Game Rule)'), ('profit_C', 'Strategy C (ML-Aware)')]:
        series = sim_runs_df[s_col]
        stats_rows.append({
            'strategy': s_label,
            'mean': series.mean(),
            'std': series.std(),
            'p05 (5th %ile)': series.quantile(0.05),
            'min': series.min(),
            'median': series.median(),
            'p95 (95th %ile)': series.quantile(0.95),
            'max': series.max(),
        })
    stats_df = pd.DataFrame(stats_rows)
    print("\nProfit Distribution across 100 repeated runs:")
    print(stats_df.to_string(index=False))
    
    diff_c_a = sim_runs_df['profit_C'] - sim_runs_df['profit_A']
    diff_b_a = sim_runs_df['profit_B'] - sim_runs_df['profit_A']
    
    print("\nDistribution of Profit Difference (C - A):")
    print(diff_c_a.describe())
    
    for eps in [0.0, 0.01, 0.05]:
        c_wins = (diff_c_a > eps).sum()
        c_ties = (diff_c_a.abs() <= eps).sum()
        c_losses = (diff_c_a < -eps).sum()
        b_wins = (diff_b_a > eps).sum()
        b_ties = (diff_b_a.abs() <= eps).sum()
        b_losses = (diff_b_a < -eps).sum()
        print(f"\nThreshold epsilon = ${eps:.2f}:")
        print(f"  Strategy C vs A: Wins={c_wins}, Losses={c_losses}, Ties={c_ties} (out of {n_sims})")
        print(f"  Strategy B vs A: Wins={b_wins}, Losses={b_losses}, Ties={b_ties} (out of {n_sims})")
        
    print("\n" + "=" * 70)
    print("STEP 2: Parameters and Correlation in Simulated World")
    print("=" * 70)
    print(f"Own-price sensitivity (b): {b}")
    print(f"Cross-price sensitivity (c): {c}")
    print(f"Ratio c / b: {c / b:.6f} ({c / b * 100:.3f}% of own-price)")
    
    lp_col = 'recommended_price'
    fp_col = 'realized_follower_price'
    
    for strat in traj_df['strategy'].unique():
        sub = traj_df[traj_df['strategy'] == strat]
        r = sub[lp_col].corr(sub[fp_col])
        std_pl = sub[lp_col].std()
        std_pf = sub[fp_col].std()
        mean_pl = sub[lp_col].mean()
        mean_pf = sub[fp_col].mean()
        print(f"Strategy: {strat}")
        print(f"  Mean P_L: ${mean_pl:.3f}, Std P_L: ${std_pl:.4f}")
        print(f"  Mean P_F: ${mean_pf:.3f}, Std P_F: ${std_pf:.4f}")
        print(f"  Correlation r(P_L, P_F): {r:.4f}")
        
    all_r = traj_df[lp_col].corr(traj_df[fp_col])
    print(f"\nOverall across all trajectories: Corr(P_L, P_F) = {all_r:.4f}")

    print("\n" + "=" * 70)
    print("STEP 3: Grid Sweep over c and Follower Response Strength")
    print("=" * 70)
    
    c_grid = [0.0, 0.25 * b, 0.50 * b, 1.0 * b]
    c_names = ["0", "0.25b", "0.50b", "1.0b"]
    strengths = [0.25, 0.75, 1.50]
    
    comp_model = load_competitor_model()
    
    grid_rows = []
    n_grid_sims = 40
    
    for c_val, c_name in zip(c_grid, c_names):
        for s_val in strengths:
            profits_a = []
            profits_b = []
            profits_c = []
            
            comp_params = {
                "behavior": "adaptive",
                "response_strength": s_val,
                "match_probability": 0.85
            }
            
            for sim_idx in range(n_grid_sims):
                seed = 2000 + sim_idx
                # A
                env_a = MarketEnvironment(
                    initial_leader_price=initial_p_l,
                    initial_follower_price=initial_p_f,
                    cost=cost,
                    a=a,
                    b=b,
                    c=c_val,
                    competitor_behavior="adaptive",
                    competitor_params=comp_params,
                    demand_noise_std=2.0,
                    competitor_noise_std=0.02,
                    random_seed=seed
                )
                df_a = run_strategy_trajectory("A", env_a, n_periods=n_periods, a=a, b=b, c=c_val, seed=seed)
                profits_a.append(df_a['realized_profit'].sum())
                
                # B
                env_b = MarketEnvironment(
                    initial_leader_price=initial_p_l,
                    initial_follower_price=initial_p_f,
                    cost=cost,
                    a=a,
                    b=b,
                    c=c_val,
                    competitor_behavior="adaptive",
                    competitor_params=comp_params,
                    demand_noise_std=2.0,
                    competitor_noise_std=0.02,
                    random_seed=seed
                )
                rule_p = {"behavior": "match", "match_probability": 0.85, "response_strength": s_val}
                df_b = run_strategy_trajectory("B", env_b, n_periods=n_periods, a=a, b=b, c=c_val, rule_params=rule_p, seed=seed)
                profits_b.append(df_b['realized_profit'].sum())
                
                # C
                env_c = MarketEnvironment(
                    initial_leader_price=initial_p_l,
                    initial_follower_price=initial_p_f,
                    cost=cost,
                    a=a,
                    b=b,
                    c=c_val,
                    competitor_behavior="adaptive",
                    competitor_params=comp_params,
                    demand_noise_std=2.0,
                    competitor_noise_std=0.02,
                    random_seed=seed
                )
                df_c = run_strategy_trajectory("C", env_c, n_periods=n_periods, a=a, b=b, c=c_val, competitor_model=comp_model, seed=seed)
                profits_c.append(df_c['realized_profit'].sum())
                
            pa = np.array(profits_a)
            pb = np.array(profits_b)
            pc = np.array(profits_c)
            
            diff_ca = pc - pa
            mean_diff = float(np.mean(diff_ca))
            std_diff = float(np.std(diff_ca))
            pct_imp = float((np.mean(pc) - np.mean(pa)) / np.mean(pa) * 100.0)
            
            # Classification
            wins = int(np.sum(diff_ca > 0.10))
            ties = int(np.sum(np.abs(diff_ca) <= 0.10))
            losses = int(np.sum(diff_ca < -0.10))
            
            if mean_diff > 0.50 and wins > losses * 1.5:
                classification = "C > A"
            elif mean_diff < -0.50 and losses > wins * 1.5:
                classification = "C < A"
            else:
                classification = "C ~ A"
                
            grid_rows.append({
                "c_ratio": c_name,
                "c_val": round(c_val, 2),
                "response_strength": s_val,
                "mean_profit_A": round(float(np.mean(pa)), 2),
                "mean_profit_B": round(float(np.mean(pb)), 2),
                "mean_profit_C": round(float(np.mean(pc)), 2),
                "mean_diff_C_A": round(mean_diff, 2),
                "pct_imp_C_A": round(pct_imp, 3),
                "wins_C": wins,
                "ties_C": ties,
                "losses_C": losses,
                "classification": classification
            })
            print(f"Cell c={c_name:5s} ({c_val:5.2f}), strength={s_val:.2f} -> Mean A=${np.mean(pa):.2f}, Mean C=${np.mean(pc):.2f}, Diff=${mean_diff:+.2f} ({pct_imp:+.2f}%) | Wins={wins}, Ties={ties}, Losses={losses} => {classification}")
            
    grid_df = pd.DataFrame(grid_rows)
    print("\n--- Summary Table of Grid Sweep ---")
    print(grid_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 4: Reconcile Phase 7 Break-Even Claim")
    print("=" * 70)
    
    # Analyze detailed simulation runs per accuracy level
    # We will simulate 40 runs across accuracy levels [0.50, 0.60, 0.70, 0.80, 0.90] with symmetric noise
    acc_levels = [0.50, 0.60, 0.70, 0.80, 0.90]
    n_acc_sims = 40
    
    rng = np.random.RandomState(42)
    acc_seeds = [int(rng.randint(1, 1_000_000)) for _ in range(n_acc_sims)]
    
    acc_stats = []
    for acc in acc_levels:
        diffs = []
        profits_a_list = []
        profits_c_list = []
        for s in acc_seeds:
            env = MarketEnvironment(
                initial_leader_price=1.38,
                initial_follower_price=1.37,
                cost=0.83,
                a=165.39,
                b=79.18,
                c=0.50,
                competitor_behavior="adaptive",
                random_seed=s,
            )
            t_a = run_strategy_trajectory("A", env, n_periods=14, seed=s)
            t_c = run_strategy_trajectory("C", env, n_periods=14, competitor_model=comp_model, target_accuracy=acc, error_mode="symmetric", seed=s)
            pa = t_a['realized_profit'].sum()
            pc = t_c['realized_profit'].sum()
            diffs.append(pc - pa)
            profits_a_list.append(pa)
            profits_c_list.append(pc)
            
        diffs = np.array(diffs)
        mean_d = float(np.mean(diffs))
        std_d = float(np.std(diffs, ddof=1))
        se_d = std_d / np.sqrt(n_acc_sims)
        ci_lower = mean_d - 1.96 * se_d
        ci_upper = mean_d + 1.96 * se_d
        
        acc_stats.append({
            "target_accuracy": acc,
            "mean_profit_A": round(float(np.mean(profits_a_list)), 2),
            "mean_profit_C": round(float(np.mean(profits_c_list)), 2),
            "mean_diff (C - A)": round(mean_d, 4),
            "std_diff": round(std_d, 4),
            "95% CI on Diff": f"[{ci_lower:+.4f}, {ci_upper:+.4f}]",
            "statistically_significant": abs(mean_d) > 1.96 * se_d
        })
        
    df_acc_analysis = pd.DataFrame(acc_stats)
    print("\nDetailed Accuracy vs Profit Analysis (N=40 runs, 14 periods, c=0.50, b=79.18):")
    print(df_acc_analysis.to_string(index=False))

if __name__ == '__main__':
    run_investigation()
