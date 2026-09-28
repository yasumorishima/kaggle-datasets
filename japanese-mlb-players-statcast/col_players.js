(function() {
    const columns = {
        "mlbam_id": "MLB Advanced Media player id. Joins to pitcher / batter in the pitch files and to season_summary.csv.",
        "name": "Full name from MLB StatsAPI.",
        "name_last_first": "'Last, First' from MLB StatsAPI (the form of Savant's player_name).",
        "birth_country": "Birth country from MLB StatsAPI: Japan, except the two players with a heritage_note.",
        "birth_city": "Birth city from MLB StatsAPI.",
        "birth_date": "Birth date from MLB StatsAPI (YYYY-MM-DD).",
        "heritage_note": "'US-born, Japanese heritage' for Lars Nootbaar and Gosuke Katoh, kept from the earlier version by an explicit list; empty otherwise.",
        "primary_position": "Primary position from MLB StatsAPI at build time (P, TWP = two-way player, C, 1B, 2B, 3B, SS, LF, CF, RF, DH).",
        "bats": "Batting side from MLB StatsAPI: L, R or S.",
        "throws": "Throwing hand from MLB StatsAPI: L or R.",
        "mlb_debut_date": "MLB debut date from MLB StatsAPI (YYYY-MM-DD); may be before 2015.",
        "roles": "Which pitch files have rows for the player: pitcher, batter, or both.",
        "teams": "MLB teams the player appeared for in the seasons covered (StatsAPI abbreviation of that season), by season (within a season in StatsAPI's order), separated by /.",
        "seasons": "Comma-separated seasons 2015-2026 with MLB regular-season games.",
        "pitching_seasons": "Seasons with rows in japanese_mlb_pitching.csv (pitches thrown in the regular season).",
        "batting_seasons": "Seasons with rows in japanese_mlb_batting.csv (plate appearances in the regular season, pitchers included).",
        "in_previous_version": "yes if the player was in the earlier (February 2026) version of this dataset, including the two whose id was wrong there; no if new.",
        "previous_wrong_id": "Hisashi Iwakuma and Yuki Matsui only: the wrong id the earlier version used (461325 is Tyler Clippard, 680686 is Josiah Gray). Empty otherwise."
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
