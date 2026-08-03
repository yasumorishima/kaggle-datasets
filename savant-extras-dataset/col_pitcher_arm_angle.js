(function() {
    const columns = {
        "pitcher": "MLB Advanced Media pitcher ID.",
        "pitcher_name": "Pitcher full name (Last, First).",
        "pitch_hand": "Pitcher throwing hand (R = Right, L = Left).",
        "n_pitches": "Total number of pitches thrown.",
        "team_id": "MLB Advanced Media team ID.",
        "ball_angle": "Arm angle at release point (degrees). 0 = sidearm, 90 = overhand.",
        "relative_release_ball_x": "Horizontal release position relative to pitcher's body (feet).",
        "release_ball_z": "Vertical release height from the ground (feet).",
        "relative_shoulder_x": "Horizontal shoulder position relative to the rubber (feet).",
        "shoulder_z": "Vertical shoulder height at release (feet).",
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
