(function() {
    const columns = {
        "id": "Entity ID (league-level record).",
        "name": "Entity name (e.g., League).",
        "year": "MLB season year (2024 or 2025).",
        "pitches": "Total pitches in the dataset.",
        "knee_down_pct": "Proportion of pitches caught in one-knee-down stance.",
        "l_down_r_up_pct": "Proportion with left knee down, right knee up.",
        "r_down_l_up_pct": "Proportion with right knee down, left knee up.",
        "both_down_pct": "Proportion with both knees down.",
        "both_up_pct": "Proportion with both knees up (traditional stance).",
        "extended_leg_pct": "Proportion with one leg extended.",
        "inside_down_pct": "Proportion with inside leg down.",
        "outside_down_pct": "Proportion with outside leg down.",
        "one_knee_framing_rv": "Framing run value for one-knee stance.",
        "other_framing_rv": "Framing run value for traditional (non-one-knee) stance.",
        "one_knee_calledstr_pct": "Called strike percentage in one-knee stance.",
        "other_calledstr_pct": "Called strike percentage in traditional stance.",
        "one_knee_blocking_rv": "Blocking run value for one-knee stance.",
        "other_blocking_rv": "Blocking run value for traditional stance.",
        "one_knee_pbwp100": "Passed balls and wild pitches per 100 pitches in one-knee stance.",
        "other_pbwp100": "Passed balls and wild pitches per 100 pitches in traditional stance.",
        "one_knee_throwing_rv": "Throwing run value for one-knee stance.",
        "other_throwing_rv": "Throwing run value for traditional stance.",
        "one_knee_csaa100": "Caught stealing above average per 100 attempts in one-knee stance.",
        "other_csaa100": "Caught stealing above average per 100 attempts in traditional stance.",
        "catching_rv": "Total catching run value.",
        "one_knee_pitching_rv": "Overall pitching run value for one-knee stance.",
        "other_pitching_rv": "Overall pitching run value for traditional stance."
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
