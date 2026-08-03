(function() {
    const columns = {
        "player_id": "MLB Advanced Media player ID.",
        "player_name": "Player full name (Last, First).",
        "team_name": "Team abbreviation.",
        "start_year": "Start of the query period (season year).",
        "end_year": "End of the query period (season year).",
        "sb_attempts": "Total stolen base attempts against the catcher.",
        "catcher_stealing_runs": "Total throwing run value (runs above average on stolen base attempts).",
        "caught_stealing_above_average": "Caught stealing above average count.",
        "n_cs": "Number of runners caught stealing.",
        "rate_cs": "Caught stealing rate (n_cs / sb_attempts).",
        "est_cs_pct": "Expected caught stealing percentage based on pop time and runner speed.",
        "cs_aa_per_throw": "Caught stealing above average per throw.",
        "seasonal_runner_speed": "Average sprint speed of runners attempting to steal (ft/sec).",
        "runner_distance_from_second": "Average runner distance from 2nd base at pitch release (feet).",
        "pop_time": "Average pop time from catch to throw reaching 2nd base (seconds).",
        "exchange_time": "Average exchange time from catch to release (seconds).",
        "arm_strength": "Average throw velocity to 2nd base (mph).",
        "n_xcs_with_flight_over_xcs": "Number of expected CS where throw flight time exceeded average.",
        "n_xcs_with_exchange_over_xcs": "Number of expected CS where exchange time exceeded average.",
        "n_xcs_with_accuracy_over_xcs": "Number of expected CS where accuracy was above average.",
        "n_xcs_with_ground_other_over_xcs": "Number of expected CS with ground ball or other outcome.",
        "n_xcs_with_onfly_other_over_xcs": "Number of expected CS with on-fly other outcome.",
        "n_xcs_with_untracked_other_over_xcs": "Number of expected CS with untracked outcome.",
        "year": "MLB season year (2024 or 2025)."
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
