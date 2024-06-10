import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

def run_experiment(numPlayers, playerConfigs):
    playerConfigs_str = json.dumps(playerConfigs)
    result = subprocess.run(['python', 'AIGame.py', '--numPlayers', str(numPlayers), '--playerConfigs', playerConfigs_str], capture_output=True, text=True)
    output = result.stdout.strip()
    
    try:
        data = json.loads(output.splitlines()[-1])  # Assuming the last line of output is the JSON data
    except json.JSONDecodeError:
        print("Failed to parse JSON output")
        data = None
    
    return data

def plot_results(settlements_built, cities_built, num_experiments):
    players = [player_name for player_name in settlements_built.keys()]
    
    games = list(range(num_experiments))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot settlements built
    for player in players:
        axes[0].plot(games, settlements_built[player], label=f'Player {player}')
    axes[0].set_title('Settlements Built by Each Player')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Settlements')
    axes[0].legend()
    
    # Plot cities built
    for player in players:
        axes[1].plot(games, cities_built[player], label=f'Player {player}')
    axes[1].set_title('Cities Built by Each Player')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Cities')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def analyze_results(results, num_experiments):
    win_counts = {}
    total_turns = []
    player_points = {}

    for result in results:
        winner = result['winner']
        num_turns = result['numTurns']
        points = result['points']
        settlements = result['settlementsBuilt']
        cities = result['citiesBuilt']

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
        
        #plot settlements and cities
        plot_results(settlements, cities, num_experiments)

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
    num_experiments = 50
    results = []
    numPlayers = 3
    playerConfigs = [
        {"name": "MCTS Player1", "usePPO": "no", "exploration_param": 0.5, "strategy": "mcts"},
        {"name": "MCTS Player2", "usePPO": "yes", "exploration_param": 0.5, "strategy": "mcts"},
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
    
    analyze_results(results, num_experiments)

if __name__ == "__main__":
    main()
