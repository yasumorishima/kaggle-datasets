// Paste into browser console (F12) on the Column Description edit page of rosters.csv
(function() {
    const columns = {
        "name":       "Player's full name in English",
        "country":    "ISO country code (e.g. USA, JPN, DOM, VEN)",
        "pool":       "WBC 2026 pool assignment (e.g. A (San Juan), B (Houston))",
        "position":   "Fielding position or pitching hand (C, 1B, 2B, 3B, SS, LF, CF, RF, DH, UTL, RHP, LHP)",
        "team":       "MLB team affiliation at time of roster announcement",
        "on_40_man":  "Whether the player is on the 40-man MLB roster (true/false)",
        "role":       "Player role: batter or pitcher"
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
