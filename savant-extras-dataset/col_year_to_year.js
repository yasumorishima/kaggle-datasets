(function() {
    const columns = {
        "name": "Player full name (Last, First).",
        "entity_id": "MLB Advanced Media player ID.",
        "2015": "xwOBA for the 2015 season (empty if player was not active).",
        "2016": "xwOBA for the 2016 season.",
        "delta_2015_2016": "Change in xwOBA from 2015 to 2016.",
        "2017": "xwOBA for the 2017 season.",
        "delta_2016_2017": "Change in xwOBA from 2016 to 2017.",
        "2018": "xwOBA for the 2018 season.",
        "delta_2017_2018": "Change in xwOBA from 2017 to 2018.",
        "2019": "xwOBA for the 2019 season.",
        "delta_2018_2019": "Change in xwOBA from 2018 to 2019.",
        "2020": "xwOBA for the 2020 season (60-game COVID season).",
        "delta_2019_2020": "Change in xwOBA from 2019 to 2020.",
        "2021": "xwOBA for the 2021 season.",
        "delta_2020_2021": "Change in xwOBA from 2020 to 2021.",
        "2022": "xwOBA for the 2022 season.",
        "delta_2021_2022": "Change in xwOBA from 2021 to 2022.",
        "2023": "xwOBA for the 2023 season.",
        "delta_2022_2023": "Change in xwOBA from 2022 to 2023.",
        "2024": "xwOBA for the 2024 season.",
        "delta_2023_2024": "Change in xwOBA from 2023 to 2024.",
        "2025": "xwOBA for the 2025 season.",
        "delta_2024_2025": "Change in xwOBA from 2024 to 2025.",
        "query_year": "Season year used to query this leaderboard (2024 or 2025)."
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
