// Paste into browser console (F12) on the Column Description edit page of pitcher_summary.csv
(function() {
    const columns = {
        "mlbam_id":          "MLB Advanced Media player ID (unique identifier)",
        "player_name":       "Pitcher's full name",
        "country":           "ISO country code (e.g. USA, JPN, DOM)",
        "total_pitches":     "Total pitches thrown (MLB regular season)",
        "PA_faced":          "Total plate appearances faced",
        "K":                 "Strikeouts",
        "BB":                "Walks allowed (base on balls)",
        "HR_allowed":        "Home runs allowed",
        "H_allowed":         "Hits allowed",
        "opp_AVG":           "Opponent batting average (H / AB)",
        "opp_SLG":           "Opponent slugging percentage (TB / AB)",
        "K_pct":             "Strikeout rate as a percentage of PA faced",
        "BB_pct":            "Walk rate as a percentage of PA faced",
        "xwOBA_against":     "Expected wOBA against based on contact quality (lower = better for pitcher)",
        "avg_velo":          "Average pitch velocity in mph",
        "avg_spin_rate":     "Average spin rate in RPM",
        "pitch_type_count":  "Number of distinct pitch types thrown",
        "primary_pitch":     "Most frequently used pitch type code (FF, SL, CH, etc.)"
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
