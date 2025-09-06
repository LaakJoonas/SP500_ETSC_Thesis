#!/usr/bin/env python3

#Batch experiment runner
#Run all experiments with different configurations


from run_experiment import run_experiment_programmatic
import time
import traceback

def run_all_experiments():
    
    experiments = [
        "SP500_ONLY",
        "SP500_VIX", 
        "SP500_VIX_MACRO",
        "SP500_VIX_TECH",
        "TECH_ONLY",
        "SP500_VIX_MACRO_TECH"
    ]
    
    models = ["standard", "weighted"]
    delay_penalties = [0.3, 0.6, 0.9]
    
    results_summary = []
    
    print("=" * 80)
    print("BATCH EXPERIMENT RUNNER - S&P500 EARLY CLASSIFICATION")
    print("=" * 80)
    print(f"Running {len(experiments)} experiments × {len(models)} models × {len(delay_penalties)} delay penalties")
    print(f"Total runs: {len(experiments) * len(models) * len(delay_penalties)}")
    print("=" * 80)
    
    run_count = 0
    total_runs = len(experiments) * len(models) * len(delay_penalties)
    
    for experiment in experiments:
        for model in models:
            for delay_penalty in delay_penalties:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] Running: {experiment} with {model} CALIMERA (λ={delay_penalty})")
                
                try:
                    start_time = time.time()
                    results_dict, results_dir = run_experiment_programmatic(
                        experiment_name=experiment,
                        model_type=model,
                        delay_penalty=delay_penalty
                    )
                    end_time = time.time()
                    
                    summary = {
                        'experiment': experiment,
                        'model': model,
                        'delay_penalty': delay_penalty,
                        'accuracy': results_dict['accuracy'],
                        'earliness': results_dict['earliness'],
                        'f1_score': results_dict['f1_score'],
                        'harmonic_mean': results_dict['harmonic_mean'],
                        'cost': results_dict['cost'],
                        'runtime': end_time - start_time,
                        'results_dir': results_dir
                    }
                    results_summary.append(summary)
                    
                    print(f"Completed in {end_time - start_time:.1f}s")
                    
                except Exception as e:
                    print(f"Failed: {e}")
                    traceback.print_exc()
                    continue
    
    print("\n" + "=" * 80)
    print("BATCH EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if results_summary:
        results_summary.sort(key=lambda x: x['harmonic_mean'], reverse=True)
        
        print(f"{'Rank':<4} {'Experiment':<20} {'Model':<10} {'λ':<5} {'Acc':<6} {'Early':<6} {'HM':<6} {'Cost':<6}")
        print("-" * 80)
        
        for i, result in enumerate(results_summary, 1):
            print(f"{i:<4} {result['experiment']:<20} {result['model']:<10} {result['delay_penalty']:<5.1f} "
                  f"{result['accuracy']:<6.3f} {result['earliness']:<6.3f} {result['harmonic_mean']:<6.3f} {result['cost']:<6.3f}")
        
        best = results_summary[0]
        print(f"\nBest Performance (Harmonic Mean): {best['experiment']} with {best['model']} CALIMERA (λ={best['delay_penalty']})")
        print(f"   Harmonic Mean: {best['harmonic_mean']:.4f} | Accuracy: {best['accuracy']:.4f} | Earliness: {best['earliness']:.4f}")
        
        with open("batch_experiment_summary.txt", "w") as f:
            f.write("BATCH EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"{'Rank':<4} {'Experiment':<20} {'Model':<10} {'λ':<5} {'Acc':<6} {'Early':<6} {'HM':<6} {'Cost':<6}\n")
            f.write("-" * 80 + "\n")
            
            for i, result in enumerate(results_summary, 1):
                f.write(f"{i:<4} {result['experiment']:<20} {result['model']:<10} {result['delay_penalty']:<5.1f} "
                       f"{result['accuracy']:<6.3f} {result['earliness']:<6.3f} {result['harmonic_mean']:<6.3f} {result['cost']:<6.3f}\n")
        
        print(f"\nSummary saved to: batch_experiment_summary.txt")
    
    print(f"\nBatch experiment completed! Processed {len(results_summary)}/{total_runs} runs successfully.")

if __name__ == "__main__":
    run_all_experiments()
