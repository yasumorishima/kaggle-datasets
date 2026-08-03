(function() {
    const columns = {
        "player_id": "MLB Advanced Media player ID.",
        "entity_name": "Player full name (Last, First).",
        "team_name": "Team abbreviation.",
        "start_year": "Start of the query period (season year).",
        "end_year": "End of the query period (season year).",
        "runner_runs_tot": "Total baserunning run value (runs above average).",
        "runner_runs_XB": "Extra base advancement run value.",
        "runner_runs_SBX": "Stolen base run value.",
        "N_runner_moved": "Total number of baserunning opportunities.",
        "runner_runs_XB_swipe": "Run value from taking extra bases aggressively.",
        "runner_runs_XB_snipe": "Run value from reading the ball well and taking extra bases.",
        "runner_runs_XB_freeze": "Run value lost from being held at a base.",
        "N_runner_moved_XB": "Number of extra base advancement opportunities.",
        "runner_runs_SB2": "Run value from stolen base attempts at 2nd base.",
        "runner_runs_SB3": "Run value from stolen base attempts at 3rd base.",
        "simple_stolen_on_running_act_SB2": "Simple stolen base run value at 2nd base.",
        "simple_stolen_on_running_act_SB3": "Simple stolen base run value at 3rd base.",
        "N_runner_moved_SBX": "Number of stolen base opportunities.",
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
