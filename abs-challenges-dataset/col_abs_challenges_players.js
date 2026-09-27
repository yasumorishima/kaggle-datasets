(function() {
    const columns = {
        "challenge_type": "Who challenged on this board: batter, pitcher or catcher.",
        "player_id": "MLB Advanced Media (MLBAM) player id. Joins to Statcast batter / pitcher / fielder_2.",
        "player_name": "Player name as shown by Baseball Savant.",
        "team_abbr": "Team abbreviation at the board's level (the Triple-A club on AAA boards, the MLB club on MLB boards); one team is shown when n_teams > 1.",
        "parent_org": "Abbreviation of the MLB parent organisation (equals team_abbr on MLB boards).",
        "level": "MLB or AAA (Triple-A).",
        "year": "Season (2025 or 2026).",
        "game_type": "Added by this dataset: R = regular season, S = spring training.",
        "n_total_sample": "Savant's sample size for the player on this board; the denominator of rate_challenges.",
        "exp_chal": "Savant's expected challenges = exp_chal_gained + exp_chal_lost.",
        "exp_chal_gained": "Savant's expected overturned (won) challenges.",
        "exp_chal_lost": "Savant's expected failed (lost) challenges.",
        "exp_chal_runs_gained": "Savant's expected run value from overturned challenges.",
        "exp_chal_runs_lost": "Savant's expected run value lost on failed challenges.",
        "n_challenges": "Challenges made.",
        "n_overturns": "Challenges won (the call was overturned).",
        "net_chal_gained": "n_overturns - exp_chal_gained.",
        "net_chal_lost": "n_fails - exp_chal_lost.",
        "n_chal_reasonable": "Challenges Savant classes as reasonable.",
        "n_chal_reasonable_opps": "Savant's count of reasonable challenge opportunities.",
        "n_fails": "Challenges lost = n_challenges - n_overturns.",
        "n_chal_runs": "Run value of the challenges = n_chal_runs_gained - n_chal_runs_lost.",
        "n_chal_runs_gained": "Savant's run value gained on overturned challenges (never negative).",
        "n_chal_runs_lost": "Savant's run value lost on failed challenges (never negative).",
        "net_chal_gained_runs": "Savant's net runs on overturned challenges vs expected. Equals n_chal_runs_gained - exp_chal_runs_gained on batter boards; on most pitcher and catcher rows it does not, and Savant's formula there is not reproduced here.",
        "net_chal_lost_runs": "Savant's net runs on failed challenges vs expected. Equals n_chal_runs_lost - exp_chal_runs_lost on batter boards; on most pitcher and catcher rows it does not, and Savant's formula there is not reproduced here.",
        "net_net_runs": "net_chal_gained_runs - net_chal_lost_runs.",
        "n_strikeouts": "Savant's strikeout count on the board.",
        "n_walks": "Savant's walk count on the board.",
        "player_team": "Savant's numeric team id for team_abbr (equals parent_org_code on MLB boards).",
        "n_teams": "Number of teams the player appeared for on this board (1-4).",
        "n_levels": "Number of levels on this board (always 1).",
        "player_at_bat": "Equals player_id; filled on batter boards only.",
        "parent_org_code": "Numeric MLB team id of the parent organisation.",
        "levelCode": "Savant's raw level code: \"1\" on MLB boards, the string \"Triple-A\" on AAA boards.",
        "rate_challenges": "n_challenges / n_total_sample.",
        "exp_rate_challenges": "exp_chal / n_total_sample.",
        "exp_rate_challenges_diff": "rate_challenges - exp_rate_challenges.",
        "rate_overturns": "n_overturns / n_challenges (empty when there are no challenges).",
        "exp_rate_overturns": "exp_chal_gained / exp_chal (empty when exp_chal_gained is 0).",
        "net_net_chal": "net_chal_gained - net_chal_lost.",
        "exp_chal_runs": "exp_chal_runs_gained - exp_chal_runs_lost.",
        "rate_chal_reasonable": "n_chal_reasonable / n_challenges (empty when there are no challenges).",
        "rate_reasonable_opp_taken": "n_chal_reasonable / n_chal_reasonable_opps (empty when n_chal_reasonable_opps is 0).",
        "runs_gained_per_chal": "n_chal_runs_gained / n_challenges (empty when there are no challenges).",
        "exp_runs_gained_per_chal": "exp_chal_runs_gained / exp_chal.",
        "overturns_vs_exp": "net_net_chal - net_net_chal_against: the player's own net overturns vs expected, minus the same for challenges made against the player's side.",
        "runs_vs_exp": "net_net_runs - net_net_runs_against.",
        "net_net_chal_against_proxy": "Equals -net_net_chal_against.",
        "net_chal_gained_runs_against_proxy": "Equals -net_chal_gained_runs_against.",
        "net_net_runs_against_proxy": "Equals -net_net_runs_against.",
        "uniqueId": "Savant's row key: player_id and year joined by an underscore.",
        "pitcher": "Equals player_id; filled on pitcher boards only.",
        "pitcher_against": "Equals pitcher where filled; pitcher boards only.",
        "fielder_2": "Equals player_id; filled on catcher boards only (Statcast's catcher column).",
        "fielder_2_against": "Equals fielder_2 where filled; catcher boards only.",
        "year_1_against": "Season of the _against measures (equals year).",
        "n_total_sample_against": "As n_total_sample, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_chal_against": "As exp_chal, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_chal_gained_against": "As exp_chal_gained, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_chal_lost_against": "As exp_chal_lost, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_chal_runs_gained_against": "As exp_chal_runs_gained, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_chal_runs_lost_against": "As exp_chal_runs_lost, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_challenges_against": "As n_challenges, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_overturns_against": "As n_overturns, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "net_chal_gained_against": "As net_chal_gained, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "net_chal_lost_against": "As net_chal_lost, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_chal_reasonable_against": "As n_chal_reasonable, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_chal_reasonable_opps_against": "As n_chal_reasonable_opps, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_fails_against": "As n_fails, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_chal_runs_against": "As n_chal_runs, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_chal_runs_gained_against": "As n_chal_runs_gained, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_chal_runs_lost_against": "As n_chal_runs_lost, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "net_chal_gained_runs_against": "Net runs vs expected for challenges made against the player's side = n_chal_runs_gained_against - exp_chal_runs_gained_against.",
        "net_chal_lost_runs_against": "Net runs vs expected for challenges made against the player's side = n_chal_runs_lost_against - exp_chal_runs_lost_against.",
        "n_strikeouts_against": "As n_strikeouts, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_walks_against": "As n_walks, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_teams_against": "As n_teams, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "n_levels_against": "As n_levels, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "rate_challenges_against": "As rate_challenges, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_rate_challenges_against": "As exp_rate_challenges, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_rate_challenges_diff_against": "As exp_rate_challenges_diff, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "rate_overturns_against": "As rate_overturns, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row; also empty when there are no such challenges.",
        "exp_rate_overturns_against": "As exp_rate_overturns, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row; also empty when exp_chal_gained_against is 0.",
        "net_net_chal_against": "As net_net_chal, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "exp_chal_runs_against": "As exp_chal_runs, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "net_net_runs_against": "As net_net_runs, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row.",
        "rate_chal_reasonable_against": "As rate_chal_reasonable, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row; also empty when there are no such challenges.",
        "rate_reasonable_opp_taken_against": "As rate_reasonable_opp_taken, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row; also empty when n_chal_reasonable_opps_against is 0.",
        "runs_gained_per_chal_against": "As runs_gained_per_chal, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row; also empty when there are no such challenges.",
        "exp_runs_gained_per_chal_against": "As exp_runs_gained_per_chal, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row."
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
