(function() {
    const columns = {
        "fielder_name": "Player full name (Last, First).",
        "player_id": "MLB Advanced Media player ID.",
        "team_name": "Team name.",
        "primary_position": "Primary position number (e.g., 6 = SS, 9 = RF).",
        "primary_position_name": "Primary position name (e.g., Shortstop, Right Fielder).",
        "total_throws": "Total number of throws recorded.",
        "total_throws_1b": "Total throws recorded from 1st base.",
        "total_throws_2b": "Total throws recorded from 2nd base.",
        "total_throws_3b": "Total throws recorded from 3rd base.",
        "total_throws_ss": "Total throws recorded from shortstop.",
        "total_throws_lf": "Total throws recorded from left field.",
        "total_throws_cf": "Total throws recorded from center field.",
        "total_throws_rf": "Total throws recorded from right field.",
        "total_throws_inf": "Total throws recorded from infield positions.",
        "total_throws_of": "Total throws recorded from outfield positions.",
        "max_arm_strength": "Maximum recorded throw velocity (mph).",
        "arm_1b": "Average throw velocity from 1st base (mph).",
        "arm_2b": "Average throw velocity from 2nd base (mph).",
        "arm_3b": "Average throw velocity from 3rd base (mph).",
        "arm_ss": "Average throw velocity from shortstop (mph).",
        "arm_lf": "Average throw velocity from left field (mph).",
        "arm_cf": "Average throw velocity from center field (mph).",
        "arm_rf": "Average throw velocity from right field (mph).",
        "arm_inf": "Average throw velocity from infield positions (mph).",
        "arm_of": "Average throw velocity from outfield positions (mph).",
        "arm_overall": "Overall average throw velocity across all positions (mph).",
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
