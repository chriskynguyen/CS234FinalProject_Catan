import subprocess
import json
import numpy as np

def run_experiment(numPlayers, playerConfigs):
    playerConfigs_str = json.dumps(playerConfigs)
    result = subprocess.run(['python', 'AIGame.py', '--numPlayers', str(numPlayers), '--playerConfigs', playerConfigs_str], capture_output=True, text=True)
    output = result.stdout.strip()

    try:
        data = json.loads(output.splitlines()[-1])  # Assuming the last line of output is the JSON data
    except json.JSONDecodeError as e:
        print("Failed to parse JSON output")
        print(f"JSONDecodeError: {e}")
        data = None
    
    return data

def analyze_results(results):
    win_counts = {}
    total_turns = []
    player_points = {}

    for result in results:
        winner = result['winner']
        num_turns = result['numTurns']
        points = result['points']

        # Track win counts
        if winner not in win_counts:
            win_counts[winner] = 0
        win_counts[winner] += 1

        # Track total turns
        total_turns.append(num_turns)

        # Track player points
        for player, pts in points.items():
            if player not in player_points:
                player_points[player] = []
            player_points[player].append(pts)

    # Calculate statistics
    total_games = len(results)
    win_rate = {player: wins / total_games for player, wins in win_counts.items()}
    avg_turns = np.mean(total_turns)
    avg_points = {player: np.mean(pts) for player, pts in player_points.items()}
    std_points = {player: np.std(pts) for player, pts in player_points.items()}

    # Print statistics
    print("Win Rates:")
    for player, rate in win_rate.items():
        print(f"{player}: {rate * 100:.2f}%")

    print(f"\nAverage Number of Turns per Game: {avg_turns:.2f}")

    print("\nAverage Points per Player:")
    for player, avg in avg_points.items():
        print(f"{player}: {avg:.2f}")

    print("\nStandard Deviation of Points per Player:")
    for player, std in std_points.items():
        print(f"{player}: {std:.2f}")

def main():
    num_experiments = 100
    results = []
    numPlayers = 2
    playerConfigs = [
        {"name": "MCTS Player", "usePPO": "no", "exploration_param": 0.5, "strategy": "mcts"},
        {"name": "Heuristic Player", "usePPO": "no", "exploration_param": 0.5, "strategy": "heuristic"}
    ]
    
    for i in range(num_experiments):
        print(f"Running experiment {i+1}/{num_experiments}...")
        result = run_experiment(numPlayers, playerConfigs)
        if result:
            results.append(result)
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Experiments completed. Results saved to experiment_results.json.")

    # load the results from the JSON file for analysis
    with open('experiment_results.json', 'r') as f:
        results = json.load(f)
    
    analyze_results(results)

if __name__ == "__main__":
    main()
