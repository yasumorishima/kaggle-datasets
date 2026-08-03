(function() {
    const columns = {
        "entity_id": "MLB Advanced Media pitcher ID.",
        "entity_name": "Pitcher full name (Last, First).",
        "entity_code": "Team abbreviation for the pitcher.",
        "team_id": "MLB Advanced Media team ID.",
        "total_pitches": "Total pitches thrown.",
        "total_pitches.1": "Total pitches thrown (duplicate column from source data).",
        "total_pitches_empty": "Total pitches thrown with bases empty.",
        "median_seconds_empty": "Median seconds between pitches with bases empty.",
        "total_pitches_onbase": "Total pitches thrown with runners on base.",
        "median_seconds_empty.1": "Median seconds between pitches with runners on base.",
        "freq_hot": "Proportion of pitches thrown at a fast pace (hot tempo).",
        "freq_warm": "Proportion of pitches thrown at a moderate pace (warm tempo).",
        "freq_cold": "Proportion of pitches thrown at a slow pace (cold tempo).",
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
