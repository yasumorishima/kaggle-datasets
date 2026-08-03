(function() {
    const columns = {
        "player_id": "MLB Advanced Media player ID.",
        "player_name": "Player full name (Last, First).",
        "team_name": "Team abbreviation.",
        "start_year": "Start of the query period (season year).",
        "end_year": "End of the query period (season year).",
        "key_target_base": "Target base for stolen base attempts (e.g., All, 2B, 3B).",
        "runs_stolen_on_running_act": "Total run value from stolen base activity.",
        "n_init": "Total number of stolen base opportunities.",
        "rate_sbx": "Stolen base attempt rate.",
        "n_sb": "Number of successful stolen bases.",
        "n_cs": "Number of times caught stealing.",
        "n_pk": "Number of pickoffs.",
        "n_bk": "Number of balks.",
        "n_fb": "Number of first-pitch breaks (early jumps).",
        "n_plus": "Number of positive running outcomes.",
        "n_minus": "Number of negative running outcomes.",
        "net_act_plus": "Net run value from positive running actions.",
        "net_act_minus": "Net run value from negative running actions.",
        "r_primary_lead": "Average primary lead distance (feet).",
        "r_secondary_lead": "Average secondary lead distance (feet).",
        "r_sec_minus_prim_lead": "Difference between secondary and primary lead distance (feet).",
        "r_primary_lead_sbx": "Average primary lead distance on stolen base attempts (feet).",
        "r_secondary_lead_sbx": "Average secondary lead distance on stolen base attempts (feet).",
        "r_sec_minus_prim_lead_sbx": "Lead distance difference on stolen base attempts (feet).",
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
