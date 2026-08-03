(function() {
    const columns = {
        "year": "MLB season year (2024 or 2025).",
        "last_name, first_name": "Pitcher full name (Last, First).",
        "pitcher_id": "MLB Advanced Media pitcher ID.",
        "team_name": "Full team name.",
        "team_name_abbrev": "Team abbreviation.",
        "pitch_hand": "Pitcher throwing hand (R = Right, L = Left).",
        "avg_speed": "Average pitch velocity (mph).",
        "pitches_thrown": "Number of times this pitch type was thrown.",
        "total_pitches": "Total pitches thrown by the pitcher (all pitch types).",
        "pitches_per_game": "Average pitches per game.",
        "pitch_per": "Usage rate of this pitch type (proportion of total pitches).",
        "pitch_type": "Pitch type code (e.g., FF = 4-Seam Fastball, SL = Slider).",
        "pitch_type_name": "Pitch type full name (e.g., 4-Seam Fastball).",
        "pitcher_break_z": "Pitcher's vertical break for this pitch (inches, gravity-adjusted).",
        "league_break_z": "League average vertical break for this pitch type (inches).",
        "diff_z": "Difference from league average vertical break (pitcher_break_z - league_break_z).",
        "rise": "Rise (positive vertical break) value for this pitch (inches).",
        "pitcher_break_z_induced": "Induced vertical break (spin-driven only, gravity removed) for this pitch (inches).",
        "pitcher_break_x": "Pitcher's horizontal break for this pitch (inches).",
        "league_break_x": "League average horizontal break for this pitch type (inches).",
        "diff_x": "Difference from league average horizontal break (pitcher_break_x - league_break_x).",
        "tail": "Horizontal tail (arm-side run) value for this pitch (inches).",
        "percent_rank_diff_z": "Percentile rank for vertical break difference vs. league average.",
        "percent_rank_diff_x": "Percentile rank for horizontal break difference vs. league average."
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
