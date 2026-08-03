(function() {
    const columns = {
        "id": "MLB Advanced Media player ID.",
        "name": "Player full name (Last, First).",
        "bbe": "Total batted ball events.",
        "gb_rate": "Ground ball rate (proportion of batted balls that are ground balls).",
        "air_rate": "Air ball rate (fly balls + line drives + pop-ups combined).",
        "fb_rate": "Fly ball rate (proportion of batted balls that are fly balls).",
        "ld_rate": "Line drive rate (proportion of batted balls that are line drives).",
        "pu_rate": "Pop-up rate (proportion of batted balls that are pop-ups).",
        "pull_rate": "Pull rate (proportion of batted balls hit to pull side).",
        "straight_rate": "Straight-away rate (proportion of batted balls hit up the middle).",
        "oppo_rate": "Opposite field rate (proportion of batted balls hit to opposite field).",
        "pull_gb_rate": "Pull ground ball rate (proportion of ground balls hit to pull side).",
        "straight_gb_rate": "Straight ground ball rate (proportion of ground balls hit up the middle).",
        "oppo_gb_rate": "Opposite field ground ball rate.",
        "pull_air_rate": "Pull air ball rate (proportion of air balls hit to pull side).",
        "straight_air_rate": "Straight air ball rate.",
        "oppo_air_rate": "Opposite field air ball rate.",
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
