(function() {
    const columns = {
        "id": "MLB Advanced Media player ID.",
        "name": "Player full name (Last, First).",
        "side": "Batter handedness (R = Right, L = Left).",
        "avg_bat_speed": "Average bat speed at contact (mph).",
        "swing_tilt": "Tilt angle of the swing plane (degrees).",
        "attack_angle": "Angle of the bat at contact relative to the pitch trajectory (degrees).",
        "attack_direction": "Horizontal direction of the swing at contact (degrees). Positive = pull side.",
        "ideal_attack_angle_rate": "Rate of swings within the ideal attack angle range (0-1).",
        "avg_intercept_y_vs_plate": "Average vertical intercept distance relative to the plate (inches).",
        "avg_intercept_y_vs_batter": "Average vertical intercept distance relative to the batter (inches).",
        "avg_batter_y_position": "Average vertical position of the batter at contact (inches).",
        "avg_batter_x_position": "Average horizontal position of the batter at contact (inches).",
        "competitive_swings": "Number of competitive swings recorded.",
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
