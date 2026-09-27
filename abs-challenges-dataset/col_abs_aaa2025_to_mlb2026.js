(function() {
    const columns = {
        "challenge_type": "batter or catcher (pitchers are not in this table).",
        "player_id": "MLB Advanced Media (MLBAM) player id. Joins to Statcast batter / pitcher / fielder_2.",
        "player_name": "Player name as shown by Baseball Savant.",
        "team_abbr_aaa2025": "Triple-A 2025 regular season: Team abbreviation at the board's level (the Triple-A club on AAA boards, the MLB club on MLB boards); one team is shown when n_teams > 1.",
        "parent_org_aaa2025": "Triple-A 2025 regular season: Abbreviation of the MLB parent organisation (equals team_abbr on MLB boards).",
        "n_total_sample_aaa2025": "Triple-A 2025 regular season: Savant's sample size for the player on this board; the denominator of rate_challenges.",
        "n_challenges_aaa2025": "Triple-A 2025 regular season: Challenges made.",
        "n_overturns_aaa2025": "Triple-A 2025 regular season: Challenges won (the call was overturned).",
        "n_fails_aaa2025": "Triple-A 2025 regular season: Challenges lost = n_challenges - n_overturns. (column names refer to abs_challenges_players.csv)",
        "rate_challenges_aaa2025": "Triple-A 2025 regular season: n_challenges / n_total_sample. (column names refer to abs_challenges_players.csv)",
        "rate_overturns_aaa2025": "Triple-A 2025 regular season: n_overturns / n_challenges (empty when there are no challenges). (column names refer to abs_challenges_players.csv)",
        "exp_chal_aaa2025": "Triple-A 2025 regular season: Savant's expected challenges = exp_chal_gained + exp_chal_lost. (column names refer to abs_challenges_players.csv)",
        "exp_rate_overturns_aaa2025": "Triple-A 2025 regular season: exp_chal_gained / exp_chal (empty when exp_chal_gained is 0). (column names refer to abs_challenges_players.csv)",
        "overturns_vs_exp_aaa2025": "Triple-A 2025 regular season: net_net_chal - net_net_chal_against: the player's own net overturns vs expected, minus the same for challenges made against the player's side. (column names refer to abs_challenges_players.csv)",
        "n_chal_runs_aaa2025": "Triple-A 2025 regular season: Run value of the challenges = n_chal_runs_gained - n_chal_runs_lost. (column names refer to abs_challenges_players.csv)",
        "exp_chal_runs_aaa2025": "Triple-A 2025 regular season: exp_chal_runs_gained - exp_chal_runs_lost. (column names refer to abs_challenges_players.csv)",
        "runs_vs_exp_aaa2025": "Triple-A 2025 regular season: net_net_runs - net_net_runs_against. (column names refer to abs_challenges_players.csv)",
        "n_chal_reasonable_opps_aaa2025": "Triple-A 2025 regular season: Savant's count of reasonable challenge opportunities.",
        "rate_reasonable_opp_taken_aaa2025": "Triple-A 2025 regular season: n_chal_reasonable / n_chal_reasonable_opps (empty when n_chal_reasonable_opps is 0). (column names refer to abs_challenges_players.csv)",
        "n_strikeouts_aaa2025": "Triple-A 2025 regular season: Savant's strikeout count on the board.",
        "n_walks_aaa2025": "Triple-A 2025 regular season: Savant's walk count on the board.",
        "team_abbr_mlb2026": "MLB 2026 regular season: Team abbreviation at the board's level (the Triple-A club on AAA boards, the MLB club on MLB boards); one team is shown when n_teams > 1.",
        "parent_org_mlb2026": "MLB 2026 regular season: Abbreviation of the MLB parent organisation (equals team_abbr on MLB boards).",
        "n_total_sample_mlb2026": "MLB 2026 regular season: Savant's sample size for the player on this board; the denominator of rate_challenges.",
        "n_challenges_mlb2026": "MLB 2026 regular season: Challenges made.",
        "n_overturns_mlb2026": "MLB 2026 regular season: Challenges won (the call was overturned).",
        "n_fails_mlb2026": "MLB 2026 regular season: Challenges lost = n_challenges - n_overturns. (column names refer to abs_challenges_players.csv)",
        "rate_challenges_mlb2026": "MLB 2026 regular season: n_challenges / n_total_sample. (column names refer to abs_challenges_players.csv)",
        "rate_overturns_mlb2026": "MLB 2026 regular season: n_overturns / n_challenges (empty when there are no challenges). (column names refer to abs_challenges_players.csv)",
        "exp_chal_mlb2026": "MLB 2026 regular season: Savant's expected challenges = exp_chal_gained + exp_chal_lost. (column names refer to abs_challenges_players.csv)",
        "exp_rate_overturns_mlb2026": "MLB 2026 regular season: exp_chal_gained / exp_chal (empty when exp_chal_gained is 0). (column names refer to abs_challenges_players.csv)",
        "overturns_vs_exp_mlb2026": "MLB 2026 regular season: net_net_chal - net_net_chal_against: the player's own net overturns vs expected, minus the same for challenges made against the player's side. (column names refer to abs_challenges_players.csv)",
        "n_chal_runs_mlb2026": "MLB 2026 regular season: Run value of the challenges = n_chal_runs_gained - n_chal_runs_lost. (column names refer to abs_challenges_players.csv)",
        "exp_chal_runs_mlb2026": "MLB 2026 regular season: exp_chal_runs_gained - exp_chal_runs_lost. (column names refer to abs_challenges_players.csv)",
        "runs_vs_exp_mlb2026": "MLB 2026 regular season: net_net_runs - net_net_runs_against. (column names refer to abs_challenges_players.csv)",
        "n_chal_reasonable_opps_mlb2026": "MLB 2026 regular season: Savant's count of reasonable challenge opportunities.",
        "rate_reasonable_opp_taken_mlb2026": "MLB 2026 regular season: n_chal_reasonable / n_chal_reasonable_opps (empty when n_chal_reasonable_opps is 0). (column names refer to abs_challenges_players.csv)",
        "n_strikeouts_mlb2026": "MLB 2026 regular season: Savant's strikeout count on the board.",
        "n_walks_mlb2026": "MLB 2026 regular season: Savant's walk count on the board."
    };
    let updated = 0, skipped = 0, failed = 0;
    const headers = document.querySelectorAll('span[title]');
    console.log('Found ' + headers.length + ' columns');

    headers.forEach((header) => {
        const name = header.getAttribute('title');
        if (!columns[name]) { skipped++; return; }
        try {
            const th = header.closest('th');
            const input = th.querySelector('input[placeholder="Please enter a description"]');
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(input, columns[name]);
            input.dispatchEvent(new Event('input', {bubbles: true}));
            input.dispatchEvent(new Event('change', {bubbles: true}));
            input.dispatchEvent(new Event('blur', {bubbles: true}));
            console.log('[OK] ' + name);
            updated++;
        } catch(e) {
            console.error('[ERROR] ' + name + ': ' + e.message);
            failed++;
        }
    });

    console.log('\n=== SUMMARY ===');
    console.log('Updated: ' + updated);
    console.log('Skipped: ' + skipped);
    console.log('Failed: ' + failed);
    console.log('\n[IMPORTANT] Please review and click SAVE!');
})();
