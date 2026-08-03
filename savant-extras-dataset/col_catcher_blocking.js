(function() {
    const columns = {
        "player_id": "MLB Advanced Media player ID.",
        "player_name": "Player full name (Last, First).",
        "team_name": "Team abbreviation.",
        "start_year": "Start of the query period (season year).",
        "end_year": "End of the query period (season year).",
        "pitches": "Total pitches received.",
        "catcher_blocking_runs": "Total blocking run value (runs above average).",
        "blocks_above_average": "Blocks above average count.",
        "n_pbwp": "Actual number of passed balls and wild pitches allowed.",
        "x_pbwp": "Expected number of passed balls and wild pitches based on pitch difficulty.",
        "blocks_above_average_per_game": "Blocks above average per game played.",
        "freq_pbwp_easy": "Frequency of easy blocking chances (proportion of total pitches).",
        "freq_pbwp_medium": "Frequency of medium-difficulty blocking chances.",
        "freq_pbwp_tough": "Frequency of tough blocking chances.",
        "diff_pbwp_easy": "Run value difference on easy blocking chances.",
        "diff_pbwp_medium": "Run value difference on medium-difficulty blocking chances.",
        "diff_pbwp_tough": "Run value difference on tough blocking chances.",
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
