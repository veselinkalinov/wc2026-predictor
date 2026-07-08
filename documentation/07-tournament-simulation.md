# 07 - Tournament Simulation

## What You Will Understand After This Lesson

- How one match is simulated.
- How group standings are built.
- How knockout teams are selected.
- How dynamic Elo/form/goals updates work inside a simulated tournament.
- How Monte Carlo champion probabilities are produced.

## First Principles

Monte Carlo simulation means running a random process many times and summarizing the outcomes.

Here:

```text
simulate tournament once -> champion
repeat N times -> count champions -> probabilities
```

If Brazil wins 120 out of 1000 simulated tournaments, its simulated champion probability is 12%.

## Project-Specific Walkthrough

File: `src/models/simulate.py`

Main class: `TournamentSimulator`

Inputs:

- Hardcoded World Cup groups.
- Host teams: United States, Mexico, Canada.
- `MatchPredictor` for scoreline probabilities and team states.

Output:

- Champion name for one tournament.
- Champion probability series for many tournaments.
- Detailed group/knockout payload for interactive UI.

## Match Simulation

`_simulate_match(team_a, team_b, is_knockout=False)`:

1. Determines home/away context:
   - If one team is a host, that team becomes home and `is_neutral=0`.
   - Otherwise the match is neutral.
2. Calls `predictor.predict_scoreline`.
3. Samples a scoreline from the scoreline probability matrix.
4. Converts score to result `H`, `D`, or `A`.
5. If knockout and tied:
   - Samples extra-time goals.
   - If still tied, uses Elo-weighted shootout probability.
6. Updates team states.
7. Returns score and winner in original team order.

Current predictor-context note:

`_simulate_match` calls `predictor.predict_scoreline` without manually supplying rest-day overrides. In the current predictor implementation, missing rest values are resolved by the predictor's context logic. The simulator then applies its own dynamic state updates after each simulated match. In an interview, describe this as two layers:

- the predictor supplies the pre-match probability/scoreline distribution
- the simulator mutates temporary tournament team state after the sampled result

Do not claim the simulator writes those simulated updates back to `feature_matrix.csv`; they are in-memory simulation state.

## Dynamic State Updates

`_update_stats_after_match` updates:

- Elo ratings.
- Form.
- Goals scored average.
- Goals conceded average.
- Goal difference average.

This matters because a tournament is sequential. A team that wins group matches can enter knockouts with updated form and Elo.

## Group Stage

`simulate_group_stage`:

- Loops through each group.
- Simulates every pair of teams once.
- Updates:
  - points
  - goal difference
  - goals for
- Sorts by points, then goal difference, then goals for.

Each group has 4 teams, so each group has 6 matches.

## Knockout Qualification

`_get_knockout_bracket_teams`:

- Takes top two from each group.
- Takes best eight third-place teams.
- Assigns third-place teams into bracket slots with hardcoded group requirements.

This represents the 48-team World Cup structure, but it is still hardcoded project logic.

## Monte Carlo

`run_monte_carlo(n_sims)`:

1. Clears prediction cache.
2. Disables predictor hot reload during simulation.
3. Spawns a multiprocessing pool.
4. Each worker creates a simulator.
5. Each simulation returns a champion.
6. Counts champions with pandas.
7. Returns probabilities.

## Known Implementation Detail

`_simulate_match_fast` currently starts with:

```python
return self._simulate_match(team_a, team_b, is_knockout=is_knockout)
```

That means the optimized NumPy code below that return is unreachable. Any interview answer should describe current behavior, not intended behavior.

## Common Interview Questions

| Question | Strong answer |
|---|---|
| What is Monte Carlo simulation? | Repeating a probabilistic process many times and estimating outcome probabilities from frequencies. |
| Why update Elo/form/goals during simulation? | Tournament performance should affect later matches in the same simulated tournament. |
| How are knockout draws resolved? | Extra time goals are sampled; if still tied, an Elo-weighted shootout picks a winner. |
| What is a limitation of this simulator? | Groups and bracket rules are hardcoded, and the fast path currently contains unreachable code. |
| Are simulated team updates persisted to data files? | No. They update in-memory team state inside the simulation run; they do not rewrite the training feature matrix. |

## Rebuild Exercise

Implement a tiny four-team group simulator:

1. Define teams A, B, C, D.
2. Randomly simulate each pair.
3. Award points.
4. Sort by points.

Then replace random match outcomes with probabilities from a dummy predictor.

## Self-Check Quiz

1. Which file defines `GROUPS`?
2. Which class runs Monte Carlo?
3. How many teams advance from third-place groups?
4. What is the current issue with `_simulate_match_fast`?

Answers:

1. `src/models/simulate.py`
2. `TournamentSimulator`
3. Eight.
4. It returns before reaching the optimized code.

## External Links

- NumPy random sampling: https://numpy.org/doc/stable/reference/random/index.html
- pandas `value_counts`: https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
