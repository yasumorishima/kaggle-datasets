(function() {
    const columns = {
        "player": "Player full name (Last, First).",
        "player_id": "MLB Advanced Media player ID.",
        "team_abbrev": "Team abbreviation.",
        "year": "MLB season year (2024 or 2025).",
        "type": "HR classification type used (e.g., adj_xhr = adjusted expected home runs).",
        "avg_hr_trot": "Average home run trot time (seconds from contact to home plate).",
        "doubters": "Number of home runs that were in doubt (not clearly going out).",
        "mostly_gone": "Number of home runs that were mostly certain to be home runs.",
        "no_doubters": "Number of no-doubt home runs (clearly over the fence by a large margin).",
        "no_doubter_per": "Percentage of home runs that were no-doubters.",
        "hr_total": "Total home runs hit.",
        "xhr": "Expected home runs based on exit velocity, launch angle, and spray angle.",
        "xhr_diff": "Difference between actual HR total and expected HR total (hr_total - xhr)."
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
