(function() {
    const columns = {
        "entity_id": "MLB Advanced Media team ID.",
        "entity_name": "Team name.",
        "year": "MLB season year (2024 or 2025).",
        "pitches": "Total pitches thrown by the team.",
        "all_violations": "Total pitch clock violations (all types combined).",
        "pitcher_timer": "Number of pitcher pitch clock violations.",
        "batter_timer": "Number of batter pitch clock violations.",
        "batter_timeout": "Number of batter timeout violations.",
        "catcher_timer": "Number of catcher pitch clock violations.",
        "defensive_shift": "Number of defensive shift violations."
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
