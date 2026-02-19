// Paste into browser console (F12) on the Column Description edit page of batter_summary.csv
(function() {
    const columns = {
        "mlbam_id":          "MLB Advanced Media player ID (unique identifier)",
        "player_name":       "Player's full name",
        "country":           "ISO country code (e.g. USA, JPN, DOM)",
        "PA":                "Plate appearances",
        "AB":                "At-bats",
        "H":                 "Hits",
        "1B":                "Singles",
        "2B":                "Doubles",
        "3B":                "Triples",
        "HR":                "Home runs",
        "BB":                "Walks (base on balls)",
        "HBP":               "Hit by pitch",
        "K":                 "Strikeouts",
        "TB":                "Total bases",
        "AVG":               "Batting average (H / AB)",
        "OBP":               "On-base percentage ((H + BB + HBP) / PA)",
        "SLG":               "Slugging percentage (TB / AB)",
        "OPS":               "On-base plus slugging (OBP + SLG)",
        "K_pct":             "Strikeout rate as a percentage of PA",
        "BB_pct":            "Walk rate as a percentage of PA",
        "xwOBA":             "Expected weighted on-base average based on launch speed and angle (Statcast)",
        "avg_exit_velo":     "Average exit velocity on batted balls in mph",
        "avg_launch_angle":  "Average launch angle on batted balls in degrees"
    };

    let updated = 0, skipped = 0, failed = 0;
    const headers = document.querySelectorAll('span[title]');
    console.log(`Found ${headers.length} columns`);

    headers.forEach(header => {
        const col = header.getAttribute('title');
        if (!columns[col]) { skipped++; return; }
        try {
            const input = header.closest('th').querySelector('input[placeholder="Please enter a description"]');
            if (!input) { failed++; return; }
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(input, columns[col]);
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            input.dispatchEvent(new Event('blur', { bubbles: true }));
            console.log(`[OK] ${col}`);
            updated++;
        } catch(e) { console.error(`[ERROR] ${col}: ${e.message}`); failed++; }
    });

    console.log(`\nUpdated: ${updated} / Skipped: ${skipped} / Failed: ${failed}`);
    console.log('[IMPORTANT] Click SAVE manually!');
})();
