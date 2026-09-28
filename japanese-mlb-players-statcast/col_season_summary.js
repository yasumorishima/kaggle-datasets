(function() {
    const columns = {
        "mlbam_id": "MLB Advanced Media player id (joins to players.csv and the pitch files).",
        "name": "Full name from MLB StatsAPI.",
        "season": "Season.",
        "role": "pitcher (a row of japanese_mlb_pitching.csv) or batter (japanese_mlb_batting.csv). Two-way seasons have one row per role.",
        "teams": "MLB teams in the season (StatsAPI abbreviations), in StatsAPI's order, separated by /.",
        "games": "MLB StatsAPI regular-season games played (pitching games for role pitcher, batting games for role batter).",
        "statsapi_pitches": "MLB StatsAPI regular-season pitches thrown (pitcher) or seen (batter).",
        "statcast_rows": "Rows of this player-season-role in the pitch file. Minus the automatic_ball / automatic_strike rows it equals statsapi_pitches (the build checks this).",
        "bat_pa": "Role batter: MLB StatsAPI regular-season plate appearances. Empty on pitcher rows.",
        "bat_ab": "Role batter: MLB StatsAPI regular-season at-bats. Empty on pitcher rows.",
        "bat_h": "Role batter: MLB StatsAPI regular-season hits. Empty on pitcher rows.",
        "bat_hr": "Role batter: MLB StatsAPI regular-season home runs. Empty on pitcher rows.",
        "bat_bb": "Role batter: MLB StatsAPI regular-season walks. Empty on pitcher rows.",
        "bat_so": "Role batter: MLB StatsAPI regular-season strikeouts. Empty on pitcher rows.",
        "bat_sb": "Role batter: MLB StatsAPI regular-season stolen bases. Empty on pitcher rows.",
        "bat_avg": "Role batter: MLB StatsAPI regular-season batting average. Empty on pitcher rows.",
        "bat_obp": "Role batter: MLB StatsAPI regular-season on-base percentage. Empty on pitcher rows.",
        "bat_slg": "Role batter: MLB StatsAPI regular-season slugging percentage. Empty on pitcher rows.",
        "bat_ops": "Role batter: MLB StatsAPI regular-season OPS. Empty on pitcher rows.",
        "bat_woba": "Role batter: wOBA from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on pitcher rows.",
        "bat_wrc_plus": "Role batter: wRC+ from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on pitcher rows.",
        "bat_war": "Role batter: WAR from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on pitcher rows.",
        "pit_gs": "Role pitcher: MLB StatsAPI regular-season games started. Empty on batter rows.",
        "pit_ip": "Role pitcher: MLB StatsAPI regular-season innings pitched (x.1 and x.2 are thirds). Empty on batter rows.",
        "pit_bf": "Role pitcher: MLB StatsAPI regular-season batters faced. Empty on batter rows.",
        "pit_w": "Role pitcher: MLB StatsAPI regular-season wins. Empty on batter rows.",
        "pit_l": "Role pitcher: MLB StatsAPI regular-season losses. Empty on batter rows.",
        "pit_sv": "Role pitcher: MLB StatsAPI regular-season saves. Empty on batter rows.",
        "pit_hld": "Role pitcher: MLB StatsAPI regular-season holds. Empty on batter rows.",
        "pit_so": "Role pitcher: MLB StatsAPI regular-season strikeouts. Empty on batter rows.",
        "pit_bb": "Role pitcher: MLB StatsAPI regular-season walks. Empty on batter rows.",
        "pit_hr": "Role pitcher: MLB StatsAPI regular-season home runs allowed. Empty on batter rows.",
        "pit_era": "Role pitcher: MLB StatsAPI regular-season ERA. Empty on batter rows.",
        "pit_whip": "Role pitcher: MLB StatsAPI regular-season WHIP. Empty on batter rows.",
        "pit_fip": "Role pitcher: FIP from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on batter rows.",
        "pit_xfip": "Role pitcher: xFIP from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on batter rows.",
        "pit_war": "Role pitcher: WAR from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on batter rows."
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
