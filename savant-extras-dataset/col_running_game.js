(function() {
    const columns = {
        "player_id": "MLB Advanced Media pitcher ID.",
        "player_name": "Pitcher full name (Last, First).",
        "team_name": "Team abbreviation.",
        "start_year": "Start of the query period (season year).",
        "end_year": "End of the query period (season year).",
        "key_target_base": "Target base for stolen base attempts tracked (e.g., All, 2B, 3B).",
        "runs_prevented_on_running_attr": "Total runs prevented on running attributes by the pitcher.",
        "n_pitcher_cs_aa": "Pitcher's caught stealing above average count.",
        "n_init": "Total number of stolen base opportunities faced.",
        "rate_sbx": "Stolen base attempt rate against this pitcher.",
        "n_sb": "Number of successful stolen bases allowed.",
        "n_cs": "Number of runners caught stealing (pitcher-assisted).",
        "n_pk": "Number of pickoffs by the pitcher.",
        "n_bk": "Number of balks by the pitcher.",
        "n_fb": "Number of first-pitch breaks (early jumps by runners).",
        "n_plus": "Number of positive outcomes (pitcher prevented a stolen base).",
        "n_minus": "Number of negative outcomes (stolen base allowed).",
        "net_attr_plus": "Net run value from positive pitching-side running game outcomes.",
        "net_attr_minus": "Net run value from negative pitching-side running game outcomes.",
        "r_primary_lead": "Average primary lead distance allowed by the pitcher (feet).",
        "r_secondary_lead": "Average secondary lead distance allowed (feet).",
        "r_sec_minus_prim_lead": "Difference between secondary and primary lead allowed (feet).",
        "r_primary_lead_sbx": "Average primary lead on stolen base attempts (feet).",
        "r_secondary_lead_sbx": "Average secondary lead on stolen base attempts (feet).",
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
